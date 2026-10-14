"""
desktop_app.py - runs the chatbot as a Mac or Windows app that updates itself.

The app starts the same small server as webchat.py, open only to this
computer and on whichever port is free, then shows the chat page in a window
of its own. The brain is inside the app, so chatting works offline; the
weather, news and lookup tools still need the internet.

Updates (see updater.py) are kept in the app's own storage folder. Each time
the app opens, it runs the newest version it has that works: an installed
update if there is one, otherwise the version it was built with.

GitHub turns this file into the downloadable apps (see "Make the Mac and
Windows apps" in README.md). To try the app window on your Mac first:

    pip3 install pywebview
    python3 desktop_app.py
"""
import importlib
import importlib.util
import os
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer

APP_NAME = "Tiny Chatbot"
MODULES = ("tools", "tinychat", "updater", "webchat")    # in the order they need each other

# A double-clicked app has nowhere to print to, so send printing nowhere.
for stream in ("stdout", "stderr"):
    if getattr(sys, stream) is None:
        setattr(sys, stream, open(os.devnull, "w"))
sys.dont_write_bytecode = True

# Apps carry their own list of trusted websites (from the certifi package),
# so secure sites like Wikipedia open on any computer.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


def storage_folder():
    """Where this app keeps its updates: the usual place on each system."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_NAME)


def version_in(folder):
    try:
        with open(os.path.join(folder, "version.txt")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


class App:
    """Which version of the chatbot is running, and switching to a newer one."""

    def __init__(self):
        # The folder holding the version the app was built with.
        self.bundle = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        self.storage = storage_folder()
        self.installed_folder = os.path.join(self.storage, "installed")
        self.previous_folder = os.path.join(self.storage, "previous")     # the update before, kept as a spare
        self.version = 0          # the version running now
        self.ready = None         # a newer version waiting to be switched to
        self.server = None
        self.switching = threading.Lock()

    def installed_version(self):
        return version_in(self.installed_folder)

    def bad_versions(self):
        try:
            with open(os.path.join(self.storage, "bad_versions.txt")) as f:
                return {int(line) for line in f if line.strip()}
        except (OSError, ValueError):
            return set()

    def mark_bad(self, version):
        os.makedirs(self.storage, exist_ok=True)
        with open(os.path.join(self.storage, "bad_versions.txt"), "a") as f:
            f.write(f"{version}\n")

    def note(self, text):
        """Keep a short log in the storage folder, for working out problems."""
        try:
            os.makedirs(self.storage, exist_ok=True)
            with open(os.path.join(self.storage, "log.txt"), "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass

    def load_from(self, folder):
        """Run the chatbot's code from an update folder."""
        for name in MODULES:
            path = os.path.join(folder, name + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        sys.modules["tools"].TIMEZONE = None      # on someone's own computer, use their clock
        self.version = version_in(folder)

    def load_built_in(self):
        """Run the chatbot's code that came inside the app."""
        for name in MODULES:
            sys.modules.pop(name, None)
        for name in MODULES:
            importlib.import_module(name)
        sys.modules["tools"].TIMEZONE = None
        self.version = version_in(self.bundle)

    def load_newest(self):
        """Start the newest version that works: the update, the spare, or the built-in one."""
        for folder in (self.installed_folder, self.previous_folder):
            version = version_in(folder)
            if version > version_in(self.bundle) and version not in self.bad_versions():
                try:
                    return self.load_from(folder)
                except Exception as error:
                    self.note(f"version {version} would not start: {type(error).__name__}: {error}")
                    self.mark_bad(version)
        self.load_built_in()

    def handler(self):
        return sys.modules["updater"].app_handler(sys.modules["webchat"].Handler, self)

    def page_file(self):
        return sys.modules["webchat"].PAGE_FILE

    def switch_to_installed(self):
        """Called by the "Update now" button. Returns True if it worked."""
        with self.switching:
            target = self.installed_version()
            if target <= self.version:
                return True
            running = {name: sys.modules.get(name) for name in MODULES}
            try:
                self.load_from(self.installed_folder)
            except Exception as error:
                self.note(f"version {target} would not start: {type(error).__name__}: {error}")
                self.mark_bad(target)
                sys.modules.update(running)               # carry on with what was running
                self.ready = None
                return False
            self.server.RequestHandlerClass = self.handler()
            self.ready = None
            return True


def main():
    app = App()
    app.load_newest()

    app.server = ThreadingHTTPServer(("127.0.0.1", 0), app.handler())   # 0 means any free port
    address = f"http://127.0.0.1:{app.server.server_address[1]}/"
    threading.Thread(target=app.server.serve_forever, daemon=True).start()
    print(f"{APP_NAME} version {app.version} is running at {address}", flush=True)

    sys.modules["updater"].start_checking(app)

    if not open_window(app, address):
        # No app window on this computer: show the chat in the web browser
        # instead, and keep answering until the program is stopped.
        webbrowser.open(address)
        threading.Event().wait()
    app.server.shutdown()


def open_window(app, address):
    """Show the chat in a window of its own. Returns False if that is not
    possible on this computer, for example when Windows refuses to load the
    window's helper files."""
    try:
        import webview
        webview.create_window(APP_NAME, address, width=560, height=820, min_size=(360, 480))
        webview.start()                 # returns when the window is closed
        return True
    except BaseException as error:      # includes errors raised while loading the window's parts
        if isinstance(error, KeyboardInterrupt):
            raise
        app.note(f"app window could not open, using the browser: {type(error).__name__}: {error}")
        return False


if __name__ == "__main__":
    main()
