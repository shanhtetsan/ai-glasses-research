import os
import re
import unittest

from starlette.requests import Request

import app_main


def _request(path, token=None, query="", host="ai-glasses-for-research.fly.dev"):
    headers = [(b"host", host.encode("utf-8"))]
    if token is not None:
        headers.append((b"x-trial-token", token.encode("utf-8")))
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query.encode("utf-8"),
        "headers": headers,
        "client": ("test", 1),
        "server": ("test", 443),
    })


class FieldTestPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_token = os.environ.get("TRIAL_TOKEN")
        os.environ["TRIAL_TOKEN"] = "viewer-secret"

    async def asyncTearDown(self):
        if self.old_token is None:
            os.environ.pop("TRIAL_TOKEN", None)
        else:
            os.environ["TRIAL_TOKEN"] = self.old_token

    async def test_requires_token(self):
        page = await app_main.field_test_page(_request("/test"))
        self.assertEqual(page.status_code, 401)
        wrong = await app_main.field_test_page(_request("/test", "nope"))
        self.assertEqual(wrong.status_code, 401)

    async def test_fails_closed_when_token_unset(self):
        os.environ.pop("TRIAL_TOKEN", None)
        page = await app_main.field_test_page(_request("/test", "viewer-secret"))
        self.assertEqual(page.status_code, 401)

    async def test_token_query_sets_cookie_and_redirects_to_clean_path(self):
        redirect = await app_main.field_test_page(_request("/test", query="token=viewer-secret"))
        self.assertEqual(redirect.status_code, 303)
        self.assertEqual(redirect.headers["location"], "/test")
        self.assertIn("trial_token=viewer-secret", redirect.headers["set-cookie"])
        self.assertIn("HttpOnly", redirect.headers["set-cookie"])
        self.assertIn("Secure", redirect.headers["set-cookie"])

    async def test_trials_redirect_unchanged_by_shared_helper(self):
        redirect = await app_main.trial_detail_page(
            "T02_PSIM", _request("/trials/T02_PSIM", query="token=viewer-secret")
        )
        self.assertEqual(redirect.status_code, 303)
        self.assertEqual(redirect.headers["location"], "/trials/T02_PSIM")
        listing = await app_main.trial_list_page(_request("/trials", query="token=viewer-secret"))
        self.assertEqual(listing.headers["location"], "/trials")

    async def test_page_serves_versioned_assets_and_allows_sockets(self):
        page = await app_main.field_test_page(_request("/test", "viewer-secret"))
        self.assertEqual(page.status_code, 200)
        body = page.body.decode("utf-8")
        self.assertRegex(body, r'src="/static/field_test\.js\?v=\d+"')
        self.assertRegex(body, r'href="/static/field_test\.css\?v=\d+"')
        self.assertIn('type="module"', body)
        js_v = re.search(r"field_test\.js\?v=(\d+)", body).group(1)
        css_v = re.search(r"field_test\.css\?v=(\d+)", body).group(1)
        self.assertEqual(js_v, css_v)
        csp = page.headers["content-security-policy"]
        self.assertIn("connect-src 'self' wss://ai-glasses-for-research.fly.dev", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(page.headers["permissions-policy"], "camera=(self), microphone=(self)")

    async def test_odd_host_header_is_not_written_into_csp(self):
        page = await app_main.field_test_page(
            _request("/test", "viewer-secret", host="evil.com; script-src *")
        )
        self.assertEqual(page.status_code, 200)
        csp = page.headers["content-security-policy"]
        self.assertNotIn("evil", csp)
        self.assertIn("connect-src 'self';", csp)

    def test_page_ids_match_script(self):
        with open("templates/field_test.html", encoding="utf-8") as handle:
            html = handle.read()
        with open("static/field_test.js", encoding="utf-8") as handle:
            script = handle.read()
        ids = set(re.findall(r'id="([^"]+)"', html))
        used = set(re.findall(r'\$\("#([A-Za-z0-9_]+)"\)', script))
        used |= {f"f_{key}" for key in (
            "trial", "pid", "site", "cond", "task", "success_criterion", "room", "present", "absent",
        )}
        used |= {"setup", "live", "done"}
        self.assertEqual(sorted(used - ids), [])

    def test_wire_formats_match_backend(self):
        from stability_runtime import MAX_JPEG_PAYLOAD_BYTES, MSG_TYPE_CAM, PCM_20MS_BYTES_16K_MONO
        import audio_stream
        with open("static/field_test_core.mjs", encoding="utf-8") as handle:
            core = handle.read()
        self.assertIn(f"MSG_TYPE_CAM = 0x{MSG_TYPE_CAM:02x}", core)
        self.assertIn(f"MIC_CHUNK_BYTES = {PCM_20MS_BYTES_16K_MONO}", core)
        self.assertIn(f"MAX_JPEG_BYTES = {MAX_JPEG_PAYLOAD_BYTES // 1024} * 1024", core)
        self.assertIn(f"TTS_SAMPLE_RATE = {audio_stream.STREAM_SR}", core)


if __name__ == "__main__":
    unittest.main()
