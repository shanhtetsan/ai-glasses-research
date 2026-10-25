import json
import os
import unittest
from pathlib import Path

from starlette.requests import Request

import app_main
from trial_archive import TrialArchive


def _request(path, token=None, query=""):
    headers = [] if token is None else [(b"x-trial-token", token.encode("utf-8"))]
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


class TrialRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_archive = app_main.trial_archive
        self.old_token = os.environ.get("TRIAL_TOKEN")
        app_main.trial_archive = TrialArchive(Path("tests/fixtures"), None)
        os.environ["TRIAL_TOKEN"] = "viewer-secret"

    async def asyncTearDown(self):
        app_main.trial_archive = self.old_archive
        if self.old_token is None:
            os.environ.pop("TRIAL_TOKEN", None)
        else:
            os.environ["TRIAL_TOKEN"] = self.old_token

    async def test_every_data_route_rejects_missing_token(self):
        responses = [
            await app_main.trial_list_page(_request("/trials")),
            await app_main.trial_detail_page("T02_PSIM", _request("/trials/T02_PSIM")),
            await app_main.trial_list_api(_request("/api/trials")),
            await app_main.trial_detail_api("T02_PSIM", _request("/api/trials/T02_PSIM")),
            await app_main.trial_frame("T02_PSIM", "11", _request("/trials/T02_PSIM/frames/11.jpg")),
            await app_main.trial_zip("T02_PSIM", _request("/trials/T02_PSIM.zip")),
        ]
        self.assertEqual([response.status_code for response in responses], [401] * 6)

    async def test_page_matches_mock_structure_and_token_query_sets_cookie(self):
        page = await app_main.trial_list_page(_request("/trials", "viewer-secret"))
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Every test, kept and replayable", page.body)
        self.assertIn(b"Turns \xe2\x80\x94 click one for the evidence", page.body)
        self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])

        redirect = await app_main.trial_detail_page(
            "T02_PSIM", _request("/trials/T02_PSIM", query="token=viewer-secret")
        )
        self.assertEqual(redirect.status_code, 303)
        self.assertEqual(redirect.headers["location"], "/trials/T02_PSIM")
        self.assertIn("HttpOnly", redirect.headers["set-cookie"])
        self.assertNotIn("viewer-secret", redirect.headers["location"])

    async def test_routes_read_real_local_fixture_with_token(self):
        token = "viewer-secret"
        listed = await app_main.trial_list_api(_request("/api/trials", token))
        detail = await app_main.trial_detail_api("T02_PSIM", _request("/api/trials/T02_PSIM", token))
        frame = await app_main.trial_frame(
            "T02_PSIM", "11", _request("/trials/T02_PSIM/frames/11.jpg", token)
        )
        zipped = await app_main.trial_zip("T02_PSIM", _request("/trials/T02_PSIM.zip", token))
        self.assertEqual([listed.status_code, detail.status_code, frame.status_code, zipped.status_code], [200] * 4)
        self.assertEqual(frame.media_type, "image/jpeg")
        self.assertEqual(zipped.media_type, "application/zip")
        self.assertIn(b'"answer_arrived_under":"T02.PSIM.s2"', detail.body)

    async def test_unscored_fixture_is_not_reported_as_zero(self):
        token = "viewer-secret"
        listed = json.loads((await app_main.trial_list_api(_request("/api/trials", token))).body)
        row = [item for item in listed["trials"] if item["id"] == "T02_PSIM"][0]
        self.assertFalse(row["scored"])
        self.assertIsNone(row["bad_guess_rate"])
        self.assertEqual(row["turn_count"], 2)
        detail = json.loads((await app_main.trial_detail_api(
            "T02_PSIM", _request("/api/trials/T02_PSIM", token))).body)
        self.assertFalse(detail["scored"])
        # Every frame URL the page will request resolves to a real JPEG.
        for window in detail["windows"]:
            self.assertTrue(window["frames"])
            for frame in window["frames"]:
                _, _, trial_id, _, name = frame["url"].split("/")
                response = await app_main.trial_frame(
                    trial_id, name[:-4], _request(frame["url"], token))
                self.assertEqual(response.status_code, 200, frame["url"])
                self.assertTrue(response.body.startswith(b"\xff\xd8"))

    async def test_page_versions_its_assets(self):
        page = await app_main.trial_list_page(_request("/trials", "viewer-secret"))
        self.assertRegex(page.body.decode(), r'/static/trial_viewer\.js\?v=\d+"')
        self.assertRegex(page.body.decode(), r'/static/trial_viewer\.css\?v=\d+"')

    async def test_viewer_fails_closed_when_token_is_not_configured(self):
        os.environ.pop("TRIAL_TOKEN", None)
        response = await app_main.trial_list_api(_request("/api/trials"))
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
