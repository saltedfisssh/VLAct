# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Bounded synchronous websocket transport for the Piper execution adapter."""


class PiperWebsocketPolicy:
    def __init__(self, host="localhost", port=8000, *, timeout=60.0):
        import websockets.sync.client

        from starVLA.piper import msgpack_numpy

        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.connection = websockets.sync.client.connect(
            uri, compression=None, open_timeout=timeout, ping_timeout=None, max_size=32 * 1024 * 1024
        )
        try:
            self.metadata = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        result = self.connection.recv(timeout=self.timeout)
        if isinstance(result, str):
            raise RuntimeError(f"Policy server error: {result}")
        return self.codec.unpackb(result)

    def get_server_metadata(self):
        return self.metadata

    def infer(self, observation):
        self.connection.send(self.packer.pack(observation))
        return self._receive()

    def close(self):
        self.connection.close()
