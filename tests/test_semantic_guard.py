"""SDK construction/resource cleanup checks; no HTTP or model calls."""
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from typesafe_sdk.constants import DEFAULT_TIMEOUT

from voxck.semantic_guard import SemanticGuard


class SemanticGuardClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_client_keeps_sdk_proxy_behavior_and_closes(self):
        client = Mock(aclose=AsyncMock())
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}, clear=True), \
                patch("typesafe_sdk.AsyncTypeSafeClient", return_value=client) as sdk, \
                patch("httpx2.AsyncClient") as http:
            guard = SemanticGuard()
            self.assertTrue(guard.enabled)
            sdk.assert_called_once_with()
            http.assert_not_called()
            await guard.close()
            client.aclose.assert_awaited_once_with()

    async def test_explicit_proxy_is_scoped_to_sdk_and_closes(self):
        client = Mock(aclose=AsyncMock())
        transport = Mock()
        env = {"TYPESAFE_API_KEY": "test-key", "TYPESAFE_PROXY": "http://127.0.0.1:7897",
               "HTTPS_PROXY": "http://existing-proxy.example", "NO_PROXY": "localhost"}
        with patch.dict(os.environ, env, clear=True), \
                patch("typesafe_sdk.AsyncTypeSafeClient", return_value=client) as sdk, \
                patch("httpx2.AsyncClient", return_value=transport) as http:
            before = dict(os.environ)
            guard = SemanticGuard(threshold=0.5, strict=False, timeout=10.0)
            http.assert_called_once_with(proxy=env["TYPESAFE_PROXY"], trust_env=False,
                                         timeout=DEFAULT_TIMEOUT)
            sdk.assert_called_once_with(http_client=transport)
            self.assertEqual(dict(os.environ), before)
            self.assertEqual((guard.threshold, guard.strict, guard.timeout), (0.5, False, 10.0))
            await guard.close()
            client.aclose.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
