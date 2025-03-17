from typing import AsyncGenerator, Generator, TypeVar

import asyncio

T = TypeVar('T')

def run_async_generator(async_fn: Generator[AsyncGenerator[T, None], None, None]) -> Generator[T, None, None]:
    """Run an async generator function in a new event loop and yield results synchronously.
    
    Args:
        async_fn: A function that returns an async generator.
        
    Yields:
        Results from the async generator as they become available.
        
    Raises:
        Exception: Any exception raised by the async generator is propagated.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Create async generator
        async_gen = async_fn()
        
        while True:
            try:
                # Run until next result is ready
                result = loop.run_until_complete(async_gen.__anext__())
                yield result
            except StopAsyncIteration:
                break
            except Exception as e:
                # Ensure loop is cleaned up on error
                if not loop.is_closed():
                    loop.close()
                raise e
    finally:
        # Clean up pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        if not loop.is_closed():
            # Run loop one final time to execute remaining callbacks
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()