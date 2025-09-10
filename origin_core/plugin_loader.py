"""
Dynamic Plugin Loader for discovering and registering agent tools at runtime.
Scans a directory for Python modules exposing a `register(registry)` entry point.
"""

import importlib
import importlib.util
import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PluginManifest:
    """Metadata about a discovered plugin."""
    name: str
    module_path: str
    version: str = "0.0.1"
    description: str = ""
    loaded: bool = False
    error: Optional[str] = None


class PluginLoader:
    """
    Discovers and loads tool plugins from a specified directory.

    Each plugin is a Python file that exposes a `register(registry)` function.
    When loaded, the function is called with the active ToolRegistry, allowing
    the plugin to register its own tools dynamically.

    Usage:
        loader = PluginLoader("./plugins")
        manifests = loader.discover()
        loader.load_all(tool_registry)
    """

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = os.path.abspath(plugin_dir)
        self._manifests: Dict[str, PluginManifest] = {}

    def discover(self) -> List[PluginManifest]:
        """
        Scans the plugin directory for valid Python modules.

        A valid plugin is any `.py` file (excluding `__init__.py`)
        that contains a callable `register` attribute.
        """
        self._manifests.clear()

        if not os.path.isdir(self.plugin_dir):
            return []

        for filename in sorted(os.listdir(self.plugin_dir)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            name = filename[:-3]
            module_path = os.path.join(self.plugin_dir, filename)

            manifest = PluginManifest(name=name, module_path=module_path)
            self._manifests[name] = manifest

        return list(self._manifests.values())

    def load_plugin(self, name: str, registry: Any = None) -> PluginManifest:
        """
        Loads a single plugin by name and optionally registers its tools.

        Args:
            name: The plugin name (filename without .py extension).
            registry: Optional ToolRegistry instance passed to the plugin's register() function.

        Returns:
            Updated PluginManifest with load status.
        """
        manifest = self._manifests.get(name)
        if not manifest:
            raise KeyError(f"Plugin '{name}' not found. Run discover() first.")

        try:
            spec = importlib.util.spec_from_file_location(
                f"origin_plugin_{name}", manifest.module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for '{name}'")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"origin_plugin_{name}"] = module
            spec.loader.exec_module(module)

            # Extract optional metadata
            manifest.version = getattr(module, "__version__", "0.0.1")
            manifest.description = getattr(module, "__description__", "")

            # Call register() if available and registry is provided
            if registry and hasattr(module, "register") and callable(module.register):
                module.register(registry)

            manifest.loaded = True

        except Exception as exc:
            manifest.loaded = False
            manifest.error = str(exc)

        return manifest

    def load_all(self, registry: Any = None) -> List[PluginManifest]:
        """Discovers and loads all plugins, returning their manifests."""
        if not self._manifests:
            self.discover()

        results = []
        for name in list(self._manifests.keys()):
            result = self.load_plugin(name, registry)
            results.append(result)

        return results

    def get_loaded_plugins(self) -> List[PluginManifest]:
        """Returns manifests of successfully loaded plugins."""
        return [m for m in self._manifests.values() if m.loaded]

    def get_failed_plugins(self) -> List[PluginManifest]:
        """Returns manifests of plugins that failed to load."""
        return [m for m in self._manifests.values() if not m.loaded and m.error]
