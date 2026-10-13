"""
updater.py - keeps the Mac and Windows apps up to date by themselves.

Each time you run "Build desktop apps" on GitHub, the Release gets two extra
files: an update pack (tiny-update.zip) holding the brain, the tools and the
chat page, and update.json saying which version that pack is.

When the app opens, it asks GitHub for update.json. If a newer version is
out, it downloads the pack in the background, checks that it arrived intact,
and puts it in the app's own storage. A bar then appears in the chat window:
"Version N is ready. Update now". Clicking it switches the open app to the new
version in about a second. If nobody clicks, the app starts on the new version
the next time it is opened.

The app itself is never replaced, so there is nothing to install and no
security warning. One exception: a change that needs a different Python or a
new package (beyond NumPy) still needs people to download the app again.

Updates only come from the GitHub repository the app was built from, over a
secure connection. Anyone who can change that repository can change the app,
so keep your GitHub account protected with two-factor sign-in.
"""
import hashlib
import io
import json
import os
import shutil
import threading
import urllib.request
import zipfile
from http import HTTPStatus

import tools

# Everything an update pack must contain. Nothing else is ever unpacked.
PACK_FILES = ("tools.py", "tinychat.py", "webchat.py", "updater.py",
              "chat_page.html", "tinychat_weights.npz", "version.txt")
LARGEST_PACK = 50 * 1024 * 1024     # bytes; anything bigger is refused


# ----------------------------------------------------------------------------
# Checking for, downloading and installing an update
# ----------------------------------------------------------------------------
def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": tools.USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(LARGEST_PACK + 1)
    if len(data) > LARGEST_PACK:
        raise ValueError("update pack is too big")
    return data


def where_updates_come_from(app):
    """The web address of the newest Release, or None if updates are off."""
    if os.environ.get("TINY_UPDATE_URL"):              # used for testing
        return os.environ["TINY_UPDATE_URL"].rstrip("/") + "/"
    try:
        with open(os.path.join(app.bundle, "update_source.txt")) as f:
            repository = f.read().strip()              # like "your-name/your-repo"
    except OSError:
        return None                                    # not built by GitHub
    return f"https://github.com/{repository}/releases/latest/download/"


def install(data, newest, folder, spare):
    """Unpack an update pack next to the installed one, then swap them over.

    The update being replaced is kept as a spare, in case the new one fails.
    """
    incoming = folder + ".incoming"
    shutil.rmtree(incoming, ignore_errors=True)
    os.makedirs(incoming)
    with zipfile.ZipFile(io.BytesIO(data)) as pack:
        for name in PACK_FILES:                        # only the expected files
            with open(os.path.join(incoming, name), "wb") as f:
                f.write(pack.read(name))
    with open(os.path.join(incoming, "version.txt")) as f:
        if int(f.read().strip()) != newest:
            raise ValueError("update pack has the wrong version number")
    if os.path.isdir(folder):
        shutil.rmtree(spare, ignore_errors=True)
        os.rename(folder, spare)
    os.rename(incoming, folder)


def check(app):
    """Look for a newer version and get it ready. Runs in the background."""
    source = where_updates_come_from(app)
    if not source:
        return
    try:
        installed = app.installed_version()
        if installed > app.version and installed not in app.bad_versions():
            app.ready = installed       # downloaded earlier, not switched to yet
        info = json.loads(download(source + "update.json"))
        newest = int(info["version"])
        if newest <= max(app.version, installed) or newest in app.bad_versions():
            return
        data = download(source + "tiny-update.zip")
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise ValueError("update pack did not arrive intact")
        install(data, newest, app.installed_folder, app.previous_folder)
        app.ready = newest
    except Exception as error:      # no internet, no Release yet, a bad pack...
        app.note(f"update check: {type(error).__name__}: {error}")


def start_checking(app):
    threading.Thread(target=check, args=(app,), daemon=True).start()


# ----------------------------------------------------------------------------
# The update bar inside the app window
# ----------------------------------------------------------------------------
BAR_SCRIPT = r"""
(function () {
  var bar = document.createElement("div");
  bar.setAttribute("role", "status");
  bar.style.cssText = "display:none;position:sticky;top:0;z-index:10;align-items:center;gap:.75rem;" +
    "justify-content:space-between;padding:.6rem 1rem;font:inherit;font-size:.95rem;" +
    "background:var(--ink,#14213D);color:var(--paper,#EEF2F7)";
  var words = document.createElement("span");
  var button = document.createElement("button");
  button.type = "button";
  button.textContent = "Update now";
  button.style.cssText = "font:inherit;font-weight:700;border:0;border-radius:.6rem;padding:.35rem .9rem;" +
    "cursor:pointer;background:var(--marker,#FFD84D);color:#14213D";
  bar.append(words, button);
  document.body.prepend(bar);

  function show(text, offerButton) {
    words.textContent = text;
    button.style.display = offerButton ? "" : "none";
    bar.style.display = "flex";
  }

  function look() {
    fetch("/app/update").then(function (r) { return r.json(); }).then(function (state) {
      if (state.ready) { show("Version " + state.ready + " is ready.", true); }
      else { setTimeout(look, 5000); }
    }).catch(function () { setTimeout(look, 5000); });
  }

  button.addEventListener("click", function () {
    button.disabled = true;
    show("Updating…", false);
    fetch("/app/update", { method: "POST" }).then(function (r) { return r.json(); }).then(function (result) {
      if (result.ok) { location.reload(); }
      else { show("The update could not be used, so you are still on version " + result.running + ".", false); }
    }).catch(function () { show("The update could not be used. Try again after reopening the app.", false); });
  });

  look();
})();
"""


def app_handler(base, app):
    """The chat page's server, plus the update bar and its two web addresses."""

    class AppHandler(base):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(app.page_file(), encoding="utf-8") as f:
                    page = f.read()
                tag = '<script src="/app/update.js"></script>'
                page = page.replace("</body>", tag + "\n</body>") if "</body>" in page else page + tag
                self.send(HTTPStatus.OK, page, "text/html")
            elif path == "/app/update.js":
                self.send(HTTPStatus.OK, BAR_SCRIPT, "text/javascript")
            elif path == "/app/update":
                self.send_json(HTTPStatus.OK, {"running": app.version, "ready": app.ready})
            else:
                super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != "/app/update":
                return super().do_POST()
            ok = app.switch_to_installed()
            self.send_json(HTTPStatus.OK, {"ok": ok, "running": app.version})

    return AppHandler
