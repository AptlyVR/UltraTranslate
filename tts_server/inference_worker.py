"""One persistent model thread, bounded admission and cancellable pending work.

This module deliberately has no model, Torch, Flask, or audio dependencies.
"""
import queue
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import CancelledError, Future, TimeoutError
from contextlib import nullcontext


class InferenceBusy(RuntimeError):
    """The caller can retry later without changing its generation request."""

    retry_after = 1


class _Cancelled(Exception):
    pass


class PreparationCancelled(RuntimeError):
    pass


class _Preparation:
    def __init__(self):
        self.lock = threading.RLock()
        self.cancelled = False
        self.future = None
        self.finished_at = None

    def check(self):
        with self.lock:
            if self.cancelled:
                raise PreparationCancelled('Voice preparation cancelled')

    def bind(self, future):
        with self.lock:
            self.future = future
            if self.cancelled:
                future.cancel()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.future is not None:
                self.future.cancel()

    def commit(self, function):
        # Cancellation and cache insertion have one ordering, including the
        # small interval after native reference encoding has returned.
        with self.lock:
            self.check()
            return function()


class PreparationRegistry:
    """Bounded cancellation identities; also handles DELETE before POST arrives."""
    def __init__(self, capacity=64, lifetime=60.0):
        self.capacity = capacity
        self.lifetime = lifetime
        self._lock = threading.Lock()
        self._entries = OrderedDict()

    def _make_room(self):
        now = time.monotonic()
        for key, item in list(self._entries.items()):
            if item.finished_at is not None and now - item.finished_at >= self.lifetime:
                del self._entries[key]
        if len(self._entries) >= self.capacity:
            for key, item in self._entries.items():
                if item.finished_at is not None:
                    del self._entries[key]
                    break
            else:
                raise InferenceBusy('Voice preparation identities are busy')

    def begin(self, identity):
        with self._lock:
            existing = self._entries.get(identity)
            if existing is not None:
                existing.check()
                raise ValueError('preparation_id was already used; use a new unique ID')
            self._make_room()
            item = _Preparation()
            self._entries[identity] = item
            return item

    def cancel(self, identity):
        with self._lock:
            item = self._entries.get(identity)
            if item is None:
                self._make_room()
                item = _Preparation()
                item.finished_at = time.monotonic()
                self._entries[identity] = item
            item.cancel()

    def finish(self, item):
        with self._lock:
            item.finished_at = time.monotonic()


class _Admission:
    def __init__(self, owner, idle_only=False):
        self.owner = owner
        self.idle_only = idle_only
        self.claimed = False
        self.released = False

    def close(self):
        # After submission only completion/cancellation may release the slot.
        with self.owner._condition:
            if not self.claimed:
                self.owner._release(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _WorkFuture(Future):
    def __init__(self):
        super().__init__()
        self.dispatched = threading.Event()


class _Stream:
    def __init__(self):
        self.queue = queue.Queue()
        self.cancelled = threading.Event()
        self.future = None

    def send(self, item):
        while not self.cancelled.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass
        raise _Cancelled()

    def __iter__(self):
        return self

    def __next__(self):
        # close() must also wake a consumer waiting for its first/next chunk.
        while not self.cancelled.is_set():
            try:
                kind, value = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.future is not None and self.future.done():
                    if not self.future.cancelled():
                        error = self.future.exception()
                        if error is not None:
                            raise error
                    raise StopIteration
                continue
            if kind == 'end':
                raise StopIteration
            if kind == 'error':
                raise value
            return value
        raise StopIteration

    def close(self):
        self.cancelled.set()
        if self.future is not None:
            self.future.cancel()


class InferenceWorker:
    def __init__(self, max_queued=2, queue_timeout=15.0, ready_timeout=180.0):
        if isinstance(max_queued, bool) or not isinstance(max_queued, int) or max_queued < 0:
            raise ValueError('max_queued must be a nonnegative integer')
        if queue_timeout <= 0 or ready_timeout <= 0:
            raise ValueError('Inference timeouts must be positive')
        self.max_inflight = 1 + max_queued
        self.queue_timeout = float(queue_timeout)
        self.ready_timeout = float(ready_timeout)
        self._condition = threading.Condition()
        self._pending = deque()
        self._inflight = 0
        self._active = None
        self._closed = False
        self._streams = set()
        self._thread = threading.Thread(target=self._run, name='VoxCPMInference', daemon=True)
        self._thread.start()
        # Expire pending HTTP/web work even while the model is busy. This thread
        # handles only Future bookkeeping and never calls a model function.
        self._expiry_thread = threading.Thread(
            target=self._expire_pending, name='VoxCPMAdmission', daemon=True)
        self._expiry_thread.start()

    def reserve(self, *, idle_only=False):
        """Reserve before reading/decoding an HTTP generation request body."""
        with self._condition:
            if self._closed:
                raise InferenceBusy('Speech service is shutting down; retry shortly')
            if idle_only and self._inflight:
                raise InferenceBusy('Speech has priority; prepare the voice when idle')
            if self._inflight >= self.max_inflight:
                raise InferenceBusy('Speech service is busy; generation queue is full')
            self._inflight += 1
            return _Admission(self, idle_only=idle_only)

    def speech_waiting(self):
        with self._condition:
            return self._inflight > 1

    def _release(self, admission):
        # Caller holds _condition, including when invoked by a cancel callback.
        if not admission.released:
            admission.released = True
            self._inflight -= 1
            self._condition.notify_all()

    def submit(self, function, *, admission=None):
        admission = self.reserve() if admission is None else admission
        with self._condition:
            if admission.owner is not self or admission.claimed or admission.released:
                raise ValueError('Invalid or already-used inference admission')
            if self._closed:
                self._release(admission)
                raise InferenceBusy('Speech service is shutting down; retry shortly')
            if admission.idle_only and self._inflight > 1:
                self._release(admission)
                raise InferenceBusy('Speech has priority; prepare the voice when idle')
            future = _WorkFuture()
            admission.claimed = True
            work = (future, function, admission, time.monotonic() + self.queue_timeout)
            self._pending.append(work)

            def cancelled(done):
                if not done.cancelled():
                    return
                with self._condition:
                    try:
                        self._pending.remove(work)
                    except ValueError:
                        pass  # The owner thread already dispatched this work.
                    else:
                        self._release(admission)
                    done.dispatched.set()

            future.add_done_callback(cancelled)
            self._condition.notify_all()
            return future

    def _expire_pending(self):
        while True:
            expired = []
            with self._condition:
                if self._closed:
                    return
                now = time.monotonic()
                for work in list(self._pending):
                    future, _, admission, deadline = work
                    if deadline > now:
                        continue
                    self._pending.remove(work)
                    claimed = future.set_running_or_notify_cancel()
                    self._release(admission)
                    future.dispatched.set()
                    if claimed:
                        expired.append(future)
                if not expired:
                    timeout = min((work[3] - now for work in self._pending), default=None)
                    self._condition.wait(timeout=timeout)
                    continue
            for future in expired:
                future.set_exception(InferenceBusy(
                    'Speech generation waited too long in the queue; retry shortly'))

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                future, function, admission, deadline = self._pending.popleft()
                if not future.set_running_or_notify_cancel():
                    self._release(admission)
                    future.dispatched.set()
                    continue
                expired = time.monotonic() >= deadline
                defer_preparation = admission.idle_only and self._inflight > 1
                if not expired and not defer_preparation:
                    self._active = future
                future.dispatched.set()
            result = None
            error = None
            try:
                if expired:
                    raise InferenceBusy('Speech generation waited too long in the queue; retry shortly')
                if defer_preparation:
                    raise InferenceBusy('Speech has priority; prepare the voice when idle')
                result = function()
            except BaseException as exc:
                error = exc
            finally:
                with self._condition:
                    self._active = None
                    self._release(admission)
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

    def snapshot(self):
        with self._condition:
            return {
                'active': self._active is not None,
                'queued': len(self._pending),
                'inflight': self._inflight,
                'capacity': self.max_inflight,
                'queue_timeout_seconds': self.queue_timeout,
                'accepting': not self._closed and self._inflight < self.max_inflight,
            }

    def generate(self, factory, lock=None, *, admission=None):
        ready = Future()
        pipe = _Stream()

        def run():
            source = None
            try:
                with lock if lock is not None else nullcontext():
                    if pipe.cancelled.is_set():
                        raise _Cancelled()
                    plan, wav, metrics, source = factory()
                    if pipe.cancelled.is_set():
                        raise _Cancelled()
                    if source is None:
                        ready.set_result((plan, wav, metrics, None))
                        return
                    ready.set_result((plan, wav, metrics, pipe))
                    for chunk in source:
                        pipe.send(('data', chunk))
                    pipe.send(('end', None))
            except _Cancelled:
                pass
            except Exception as exc:
                if not ready.done():
                    ready.set_exception(exc)
                else:
                    try:
                        pipe.send(('error', exc))
                    except _Cancelled:
                        pass
            finally:
                if source is not None and hasattr(source, 'close'):
                    source.close()

        pipe.future = self.submit(run, admission=admission)
        with self._condition:
            self._streams.add(pipe)

        def completed(future):
            with self._condition:
                self._streams.discard(pipe)
            if not ready.done():
                error = CancelledError('Speech generation was cancelled') if future.cancelled() else future.exception()
                ready.set_exception(error or CancelledError('Speech generation was cancelled'))

        pipe.future.add_done_callback(completed)
        try:
            if not pipe.future.dispatched.wait(self.queue_timeout):
                pipe.close()
                raise InferenceBusy('Speech generation waited too long in the queue; retry shortly')
            return ready.result(timeout=self.ready_timeout)
        except TimeoutError as exc:
            pipe.close()
            raise InferenceBusy('Speech preparation timed out; retry shortly') from exc
        except BaseException:
            pipe.close()
            raise

    def shutdown(self, wait=True):
        with self._condition:
            self._closed = True
            pending = [work[0] for work in self._pending]
            streams = list(self._streams)
            self._condition.notify_all()
        for future in pending:
            future.cancel()
        for pipe in streams:
            pipe.close()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join()
        if wait and threading.current_thread() is not self._expiry_thread:
            self._expiry_thread.join()
