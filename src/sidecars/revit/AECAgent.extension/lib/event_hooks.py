"""
Revit document event hooks for automatic cache synchronization.

Registers handlers for document events (open, save, close) to keep
the SQLite cache in sync with the Revit model.
"""

from pyrevit import HOST_APP
from pyrevit.coreutils import logger


# Track registered state to prevent duplicate registration
_hooks_registered = False


def on_document_opened(sender, args):
    """
    Sync cache when a document is opened.

    Args:
        sender: Event sender
        args: DocumentOpenedEventArgs with Document property
    """
    doc = args.Document
    if doc and not doc.IsFamilyDocument:
        logger.info("Document opened: {}".format(doc.Title))
        try:
            from cache_manager import RevitCache
            cache = RevitCache(doc)
            results = cache.sync_all()
            logger.info("Initial cache sync completed: {}".format(results))
        except Exception as e:
            logger.error("Cache sync failed on document open: {}".format(e))


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
        try:
            from cache_manager import RevitCache
            cache = RevitCache(doc)
            results = cache.sync_all()
            logger.info("Post-save cache sync completed: {}".format(results))
        except Exception as e:
            logger.error("Cache sync failed on document save: {}".format(e))


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
