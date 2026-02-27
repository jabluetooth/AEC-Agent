2026-02-27 15:04:46 - ERROR - mcp.client.sse - Error in sse_reader
Traceback (most recent call last):  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_transports\default.py", line 101, in map_httpcore_exceptions     yield File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_transports\default.py", line 271, in __aiter__   async for part in self._httpcore_stream:  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_async\connection_pool.py", line 407, in __aiter__ raise exc from None
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_async\connection_pool.py", line 403, in __aiter__
    async for part in self._stream:
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_async\http11.py", line 342, in __aiter__
    raise exc
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_async\http11.py", line 334, in __aiter__
    async for chunk in self._connection._receive_response_body(**kwargs):
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_async\http11.py", line 203, in _receive_response_body       
    event = await self._receive_event(timeout=timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_async\http11.py", line 217, in _receive_event
    data = await self._network_stream.read(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_backends\anyio.py", line 32, in read
    with map_exceptions(exc_map):
  File "C:\Users\frelatorre\AppData\Local\Programs\Python\Python312\Lib\contextlib.py", line 155, in __exit__
    self.gen.throw(value)
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpcore\_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadTimeout

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\mcp\client\sse.py", line 81, in sse_reader
    async for sse in event_source.aiter_sse():  # pragma: no branch
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx_sse\_api.py", line 42, in aiter_sse
    async for line in lines:
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx_sse\_api.py", line 80, in _aiter_sse_lines
    async for text in response.aiter_text():
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_models.py", line 1018, in aiter_text
    async for byte_content in self.aiter_bytes():
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_models.py", line 997, in aiter_bytes
    async for raw_bytes in self.aiter_raw():
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_models.py", line 1055, in aiter_raw
    async for raw_stream_bytes in self.stream:
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_client.py", line 176, in __aiter__
    async for chunk in self._stream:
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_transports\default.py", line 270, in __aiter__
    with map_httpcore_exceptions():
  File "C:\Users\frelatorre\AppData\Local\Programs\Python\Python312\Lib\contextlib.py", line 155, in __exit__
    self.gen.throw(value)
  File "C:\Users\frelatorre\Desktop\Projects\aec\venv\Lib\site-packages\httpx\_transports\default.py", line 118, in map_httpcore_exceptions   
    raise mapped_exc(message) from exc
httpx.ReadTimeout