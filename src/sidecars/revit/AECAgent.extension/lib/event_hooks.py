"""
Revit document event hooks for automatic cache synchronization.

Registers handlers for document events (open, save, close) to keep
the SQLite cache in sync with the Revit model.

Also triggers PostgreSQL sync for semantic search and spatial queries.
"""

import os
import threading

from pyrevit import HOST_APP
from pyrevit.coreutils import logger


# Track registered state to prevent duplicate registration
_hooks_registered = False

# MCP server configuration (for PostgreSQL sync notifications)
_MCP_SERVER_PORT = os.environ.get("MCP_SERVER_PORT", "54321")
_ENABLE_POSTGRES_SYNC = os.environ.get("ENABLE_POSTGRES_SYNC", "true").lower() == "true"


def _notify_postgres_sync(file_path, source="revit", document_title=None):
    """
    Notify MCP server to trigger PostgreSQL sync.

    Runs in background thread to avoid blocking Revit UI.

    Args:
        file_path: Path to the document
        source: 'autocad' or 'revit'
        document_title: Optional document title for logging
    """
    if not _ENABLE_POSTGRES_SYNC:
        return

    def do_notify():
        try:
            # Use urllib since requests may not be available in all pyRevit environments
            try:
                import requests
                response = requests.post(
                    "http://localhost:{}/tools/notify_file_opened".format(_MCP_SERVER_PORT),
                    json={
                        "source": source,
                        "file_path": file_path,
                        "document_title": document_title or file_path,
                        "force_sync": False
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("PostgreSQL sync notification sent successfully")
                else:
                    logger.warn("PostgreSQL sync notification failed: {}".format(response.status_code))
            except ImportError:
                # Fallback to urllib if requests not available
                import json
                try:
                    # Python 2/3 compatibility
                    from urllib2 import Request, urlopen, HTTPError
                except ImportError:
                    from urllib.request import Request, urlopen
                    from urllib.error import HTTPError

                data = json.dumps({
                    "source": source,
                    "file_path": file_path,
                    "document_title": document_title or file_path,
                    "force_sync": False
                }).encode('utf-8')

                req = Request(
                    "http://localhost:{}/tools/notify_file_opened".format(_MCP_SERVER_PORT),
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                try:
                    urlopen(req, timeout=5)
                    logger.info("PostgreSQL sync notification sent successfully")
                except HTTPError as e:
                    logger.warn("PostgreSQL sync notification failed: {}".format(e.code))

        except Exception as e:
            # Don't fail on notification errors - PostgreSQL sync is optional
            logger.debug("PostgreSQL sync notification skipped: {}".format(e))

    # Run in background thread
    thread = threading.Thread(target=do_notify)
    thread.daemon = True
    thread.start()


def on_document_opened(sender, args):
    """
    Sync cache when a document is opened.

    Syncs both SQLite cache (fast, local) and triggers PostgreSQL sync
    (for semantic search and spatial queries).

    Args:
        sender: Event sender
        args: DocumentOpenedEventArgs with Document property
    """
    doc = args.Document
    if doc and not doc.IsFamilyDocument:
        logger.info("Document opened: {}".format(doc.Title))

        # SQLite cache sync (fast, local)
        try:
            from cache_manager import RevitCache
            cache = RevitCache(doc)
            results = cache.sync_all()
            logger.info("Initial cache sync completed: {}".format(results))
        except Exception as e:
            logger.error("Cache sync failed on document open: {}".format(e))

        # PostgreSQL sync notification (background, for semantic search)
        try:
            _notify_postgres_sync(
                file_path=doc.PathName,
                source="revit",
                document_title=doc.Title
            )
        except Exception as e:
            logger.debug("PostgreSQL sync notification failed: {}".format(e))


def on_document_saved(sender, args):
    """
    Sync cache when a document is saved.

    Args:
        sender: Event sender
        args: DocumentSavedEventArgs with Document property
    """
    doc = args.Document
    if doc and not doc.IsFamilyDocument:
        logger.info("Document saved: {}".format(doc.Title))

        # SQLite cache sync
        try:
            from cache_manager import RevitCache
            cache = RevitCache(doc)
            results = cache.sync_all()
            logger.info("Post-save cache sync completed: {}".format(results))
        except Exception as e:
            logger.error("Cache sync failed on document save: {}".format(e))

        # PostgreSQL sync notification (force=True since file changed)
        try:
            _notify_postgres_sync(
                file_path=doc.PathName,
                source="revit",
                document_title=doc.Title
            )
        except Exception as e:
            logger.debug("PostgreSQL sync notification failed: {}".format(e))


def on_document_closing(sender, args):
    """
    Log when a document is closing.

    Args:
        sender: Event sender
        args: DocumentClosingEventArgs with Document property
    """
    doc = args.Document
    if doc:
        logger.info("Document closing: {}".format(doc.Title))


def on_document_synchronizing_with_central(sender, args):
    """
    Sync cache when synchronizing with central (workshared models).

    Args:
        sender: Event sender
        args: DocumentSynchronizingWithCentralEventArgs
    """
    doc = args.Document
    if doc:
        logger.info("Synchronizing with central: {}".format(doc.Title))


def on_document_synchronized_with_central(sender, args):
    """
    Sync cache after synchronizing with central.

    Args:
        sender: Event sender
        args: DocumentSynchronizedWithCentralEventArgs
    """
    doc = args.Document
    if doc:
        logger.info("Synchronized with central: {}".format(doc.Title))
        try:
            from cache_manager import RevitCache
            cache = RevitCache(doc)
            results = cache.sync_all()
            logger.info("Post-sync cache update completed: {}".format(results))
        except Exception as e:
            logger.error("Cache sync failed after central sync: {}".format(e))


def register_hooks():
    """
    Register document event hooks.

    Should be called once during extension startup.
    """
    global _hooks_registered

    if _hooks_registered:
        logger.warn("Event hooks already registered, skipping")
        return False

    try:
        app = HOST_APP.app

        # Document lifecycle events
        app.DocumentOpened += on_document_opened
        app.DocumentSaved += on_document_saved
        app.DocumentClosing += on_document_closing

        # Worksharing events (for BIM 360 / cloud models)
        try:
            app.DocumentSynchronizingWithCentral += on_document_synchronizing_with_central
            app.DocumentSynchronizedWithCentral += on_document_synchronized_with_central
            logger.info("Worksharing event hooks registered")
        except AttributeError:
            # These events may not be available in all Revit versions
            logger.warn("Worksharing events not available in this Revit version")

        _hooks_registered = True
        logger.info("Document event hooks registered successfully")
        return True

    except Exception as e:
        logger.error("Failed to register event hooks: {}".format(e))
        return False


def unregister_hooks():
    """
    Unregister document event hooks.

    Should be called during extension shutdown if needed.
    """
    global _hooks_registered

    if not _hooks_registered:
        return False

    try:
        app = HOST_APP.app

        app.DocumentOpened -= on_document_opened
        app.DocumentSaved -= on_document_saved
        app.DocumentClosing -= on_document_closing

        try:
            app.DocumentSynchronizingWithCentral -= on_document_synchronizing_with_central
            app.DocumentSynchronizedWithCentral -= on_document_synchronized_with_central
        except AttributeError:
            pass

        _hooks_registered = False
        logger.info("Document event hooks unregistered")
        return True

    except Exception as e:
        logger.error("Failed to unregister event hooks: {}".format(e))
        return False
