import os
import sys
def suppress_ffmpeg_warnings():
    """Context manager to suppress FFmpeg warnings at file descriptor level"""
    if sys.platform == 'win32':
        return contextlib.nullcontext()  # Skip on Windows
    
    import contextlib
    
    @contextlib.contextmanager
    def _suppress():
        stderr_fd = 2  # stderr file descriptor
        stderr_backup = os.dup(stderr_fd)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        
        try:
            os.dup2(null_fd, stderr_fd)
            yield
        finally:
            os.dup2(stderr_backup, stderr_fd)
            os.close(stderr_backup)
            os.close(null_fd)
    
    return _suppress()
