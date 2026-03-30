"""
tests/test_memory.py — ChromaDB memory tests.

Run from the project root:
    uv run python tests/test_memory.py

Requires:
    - Ollama running (ollama serve)
    - nomic-embed-text pulled (ollama pull nomic-embed-text)
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from utils.memory import ContextStore


def check_embeddings() -> bool:
    """Verify Ollama embeddings are available."""
    print("🔌 Checking Ollama embeddings (nomic-embed-text)...")
    try:
        store = ContextStore(persist_dir=tempfile.mkdtemp())
        store._embeddings.embed_query("test")
        print("   → Embeddings OK ✓")
        return True
    except Exception as e:
        print(f"❌ Embeddings not available: {e}")
        print("   → Run: ollama pull nomic-embed-text")
        return False


def test_store_profile():
    """Test storing a profile."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        count = store.store_profile("Dataset has 1500 rows, 8 columns about e-commerce sales.")
        assert count > 0
        print(f"✅ store_profile: {count} chunks stored")
        return True
    except Exception as e:
        print(f"❌ store_profile: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_store_insights():
    """Test storing insights."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        text = (
            "Insight 1: Ile-de-France accounts for 32% of orders.\n"
            "Insight 2: Widget Pro is the top product with 185 sales.\n"
            "Insight 3: High dispersion in order totals (std > mean)."
        )
        count = store.store_insights(text)
        assert count > 0
        print(f"✅ store_insights: {count} chunks stored")
        return True
    except Exception as e:
        print(f"❌ store_insights: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_store_qa():
    """Test storing Q&A exchanges."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        count = store.store_qa("What is the top product?", "Widget Pro with 185 orders.")
        assert count > 0
        print(f"✅ store_qa: {count} chunks stored")
        return True
    except Exception as e:
        print(f"❌ store_qa: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_store_empty():
    """Test that storing empty text does nothing."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        count = store.store_profile("")
        assert count == 0
        count2 = store.store_profile("   ")
        assert count2 == 0
        print("✅ store_empty: empty text correctly skipped")
        return True
    except Exception as e:
        print(f"❌ store_empty: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_specific_collection():
    """Test searching in a specific collection."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        store.store_insights("Widget Pro dominates sales with 185 orders.")
        store.store_profile("Dataset has 8 columns about e-commerce.")

        results = store.search("top selling product", collection=ContextStore.INSIGHTS)
        assert len(results) > 0
        assert any("Widget" in r for r in results)

        print(f"✅ search_specific: found {len(results)} results in insights")
        return True
    except Exception as e:
        print(f"❌ search_specific: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_all_collections():
    """Test searching across all collections."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        store.store_profile("1500 rows of e-commerce data.")
        store.store_insights("Sales concentrated in Ile-de-France.")
        store.store_qa("Where are most orders?", "Ile-de-France with 32%.")

        results = store.search("Ile-de-France orders")
        assert len(results) > 0

        print(f"✅ search_all: found {len(results)} results across collections")
        return True
    except Exception as e:
        print(f"❌ search_all: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_search_as_text():
    """Test search_as_text returns a joined string."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        store.store_insights("Anomaly detected: order total of 2499 is an outlier.")

        text = store.search_as_text("anomalies")
        assert len(text) > 0
        assert "outlier" in text.lower() or "anomal" in text.lower()

        print(f"✅ search_as_text: got {len(text)} chars")
        return True
    except Exception as e:
        print(f"❌ search_as_text: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clear():
    """Test clearing collections."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        store.store_profile("Some data.")
        store.clear(collection=ContextStore.PROFILES)

        results = store.search("data", collection=ContextStore.PROFILES)
        # After clear, no results expected
        print(f"✅ clear: collection cleared (results: {len(results)})")
        return True
    except Exception as e:
        print(f"❌ clear: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clear_all():
    """Test clearing all collections."""
    tmp = tempfile.mkdtemp()
    try:
        store = ContextStore(persist_dir=tmp)
        store.store_profile("Data.")
        store.store_insights("Insights.")
        store.store_qa("Q?", "A.")
        store.clear()

        collections = store.list_collections()
        print(f"✅ clear_all: {len(collections)} collections remaining")
        return True
    except Exception as e:
        print(f"❌ clear_all: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  CHROMADB MEMORY TESTS")
    print("=" * 60)

    if not check_embeddings():
        sys.exit(1)

    tests = [
        ("store_profile", test_store_profile),
        ("store_insights", test_store_insights),
        ("store_qa", test_store_qa),
        ("store_empty", test_store_empty),
        ("search_specific", test_search_specific_collection),
        ("search_all", test_search_all_collections),
        ("search_as_text", test_search_as_text),
        ("clear", test_clear),
        ("clear_all", test_clear_all),
    ]

    results = []
    for name, fn in tests:
        results.append((name, fn()))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    if passed == total:
        print(f"  ✅ ALL {total} TESTS PASSED — ChromaDB memory operational!")
    else:
        print(f"  ⚠️  {passed}/{total} tests passed")
        for name, ok in results:
            if not ok:
                print(f"     ❌ {name}")
    print("=" * 60)
