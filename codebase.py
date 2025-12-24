"""
Codebase Understanding: Semantic map of the project.

This module builds and maintains understanding of:
- File structure and purposes
- Dependencies and imports
- Key functions and classes
- Patterns and conventions
"""

import ast
import os
import re
import json
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FileInfo:
    """Information about a file in the codebase."""
    path: str
    language: str
    size_bytes: int
    line_count: int
    purpose: str = ""
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=datetime.now)
    hash: str = ""


@dataclass
class CodebaseMap:
    """A map of the entire codebase."""
    root_path: str
    files: dict[str, FileInfo] = field(default_factory=dict)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    summary: str = ""


class CodebaseAnalyzer:
    """
    Analyzes and understands codebases.

    Capabilities:
    - Parse file structure
    - Extract symbols (functions, classes, imports)
    - Build dependency graph
    - Identify patterns and conventions
    - Generate summaries
    """

    # File extensions we understand
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".sh": "bash",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }

    # Directories to ignore
    IGNORE_DIRS = {
        ".git", ".svn", ".hg",
        "node_modules", "vendor", "venv", ".venv", "env",
        "__pycache__", ".pytest_cache", ".mypy_cache",
        "dist", "build", "target", "out",
        ".idea", ".vscode", ".vs",
        "coverage", ".coverage",
    }

    # Files to ignore
    IGNORE_FILES = {
        ".gitignore", ".dockerignore",
        "package-lock.json", "yarn.lock", "Pipfile.lock",
        ".DS_Store", "Thumbs.db",
    }

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "codebase.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the codebase database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS codebases (
                id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                summary TEXT,
                patterns TEXT,
                conventions TEXT,
                entry_points TEXT,
                analyzed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                codebase_id TEXT,
                path TEXT NOT NULL,
                language TEXT,
                size_bytes INTEGER,
                line_count INTEGER,
                purpose TEXT,
                imports TEXT,
                exports TEXT,
                functions TEXT,
                classes TEXT,
                dependencies TEXT,
                hash TEXT,
                analyzed_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def analyze(self, root_path: str, force: bool = False) -> CodebaseMap:
        """
        Analyze a codebase and build a semantic map.

        Args:
            root_path: Path to the codebase root
            force: Force re-analysis even if cached

        Returns:
            CodebaseMap with analysis results
        """
        root_path = os.path.abspath(root_path)
        codebase_id = hashlib.md5(root_path.encode()).hexdigest()[:16]

        # Check cache
        if not force:
            cached = self._load_cached(codebase_id)
            if cached:
                return cached

        # Analyze all files
        codebase_map = CodebaseMap(root_path=root_path)

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Filter ignored directories
            dirnames[:] = [d for d in dirnames if d not in self.IGNORE_DIRS]

            for filename in filenames:
                if filename in self.IGNORE_FILES:
                    continue

                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, root_path)

                # Get file extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.LANGUAGE_MAP:
                    continue

                # Analyze file
                file_info = self._analyze_file(filepath, rel_path)
                if file_info:
                    codebase_map.files[rel_path] = file_info

        # Build dependency graph
        codebase_map.dependency_graph = self._build_dependency_graph(codebase_map.files)

        # Identify entry points
        codebase_map.entry_points = self._find_entry_points(codebase_map.files)

        # Detect patterns and conventions
        codebase_map.patterns = self._detect_patterns(codebase_map.files)
        codebase_map.conventions = self._detect_conventions(codebase_map.files)

        # Generate summary
        codebase_map.summary = self._generate_summary(codebase_map)

        # Cache results
        self._cache_analysis(codebase_id, codebase_map)

        return codebase_map

    def _analyze_file(self, filepath: str, rel_path: str) -> FileInfo | None:
        """Analyze a single file."""
        try:
            stat = os.stat(filepath)
            ext = os.path.splitext(filepath)[1].lower()
            language = self.LANGUAGE_MAP.get(ext, "unknown")

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')
            line_count = len(lines)

            # Calculate hash
            file_hash = hashlib.md5(content.encode()).hexdigest()[:16]

            # Extract symbols based on language
            imports = []
            exports = []
            functions = []
            classes = []

            if language == "python":
                imports, exports, functions, classes = self._analyze_python(content)
            elif language in ("javascript", "typescript"):
                imports, exports, functions, classes = self._analyze_javascript(content)
            else:
                # Generic extraction
                functions = re.findall(r'(?:def|function|func|fn)\s+(\w+)', content)
                classes = re.findall(r'class\s+(\w+)', content)

            # Infer purpose from path and content
            purpose = self._infer_purpose(rel_path, content, functions, classes)

            return FileInfo(
                path=rel_path,
                language=language,
                size_bytes=stat.st_size,
                line_count=line_count,
                purpose=purpose,
                imports=imports[:50],  # Limit
                exports=exports[:50],
                functions=functions[:50],
                classes=classes[:20],
                dependencies=[],  # Will be filled by dependency graph
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                hash=file_hash,
            )

        except Exception as e:
            return None

    def _analyze_python(self, content: str) -> tuple[list, list, list, list]:
        """Analyze Python code."""
        imports = []
        exports = []
        functions = []
        classes = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    if not node.name.startswith('_'):
                        exports.append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    functions.append(node.name)
                    if not node.name.startswith('_'):
                        exports.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    if not node.name.startswith('_'):
                        exports.append(node.name)

        except SyntaxError:
            # Fallback to regex
            imports = re.findall(r'^(?:from|import)\s+(\w+)', content, re.MULTILINE)
            functions = re.findall(r'^(?:async\s+)?def\s+(\w+)', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)

        return imports, exports, functions, classes

    def _analyze_javascript(self, content: str) -> tuple[list, list, list, list]:
        """Analyze JavaScript/TypeScript code."""
        imports = re.findall(r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]", content)
        exports = re.findall(r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)", content)
        functions = re.findall(r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s*)?\(|[=:]\s*(?:async\s+)?function)", content)
        classes = re.findall(r"class\s+(\w+)", content)

        return imports, exports, functions, classes

    def _infer_purpose(
        self,
        rel_path: str,
        content: str,
        functions: list[str],
        classes: list[str],
    ) -> str:
        """Infer the purpose of a file."""
        path_lower = rel_path.lower()
        filename = os.path.basename(rel_path).lower()

        # Check path patterns
        if "test" in path_lower:
            return "Test file"
        if "config" in filename or filename in ("settings.py", "config.py", ".env"):
            return "Configuration"
        if filename in ("__init__.py", "index.js", "index.ts"):
            return "Module entry point"
        if filename in ("main.py", "main.js", "main.go", "app.py"):
            return "Application entry point"
        if filename in ("readme.md", "readme.txt"):
            return "Documentation"
        if filename in ("setup.py", "pyproject.toml", "package.json", "cargo.toml"):
            return "Project configuration"
        if "util" in path_lower or "helper" in path_lower:
            return "Utility functions"
        if "model" in path_lower:
            return "Data models"
        if "route" in path_lower or "api" in path_lower:
            return "API routes"
        if "component" in path_lower:
            return "UI component"

        # Infer from content
        if classes and "Agent" in str(classes):
            return "Agent implementation"
        if "router" in str(functions).lower():
            return "Routing logic"
        if "parse" in str(functions).lower() or "Parser" in str(classes):
            return "Parser/processor"

        return "Source file"

    def _build_dependency_graph(self, files: dict[str, FileInfo]) -> dict[str, list[str]]:
        """Build a dependency graph between files."""
        graph = {}

        for path, info in files.items():
            deps = []

            # Match imports to files
            for imp in info.imports:
                # Convert import to potential file paths
                potential_paths = [
                    f"{imp.replace('.', '/')}.py",
                    f"{imp.replace('.', '/')}/__init__.py",
                    f"{imp}.py",
                    imp,
                ]

                for pp in potential_paths:
                    if pp in files:
                        deps.append(pp)
                        break

            graph[path] = deps

        return graph

    def _find_entry_points(self, files: dict[str, FileInfo]) -> list[str]:
        """Find entry points in the codebase."""
        entry_points = []

        for path, info in files.items():
            filename = os.path.basename(path).lower()

            # Common entry point patterns
            if filename in ("main.py", "app.py", "cli.py", "main.go", "main.rs"):
                entry_points.append(path)
            elif filename.endswith("__main__.py"):
                entry_points.append(path)
            elif "if __name__" in str(info.functions):
                entry_points.append(path)

            # Check for CLI decorators/patterns
            if "click" in info.imports or "argparse" in info.imports:
                if path not in entry_points:
                    entry_points.append(path)

        return entry_points

    def _detect_patterns(self, files: dict[str, FileInfo]) -> list[str]:
        """Detect architectural patterns in the codebase."""
        patterns = []

        # Check for common patterns
        paths = set(files.keys())
        path_str = " ".join(paths)

        if "agents/" in path_str or "agent" in path_str:
            patterns.append("Agent pattern")
        if "models/" in path_str or "model" in path_str:
            patterns.append("MVC/Model layer")
        if "views/" in path_str or "templates/" in path_str:
            patterns.append("MVC/View layer")
        if "controllers/" in path_str:
            patterns.append("MVC/Controller layer")
        if "routes/" in path_str or "api/" in path_str:
            patterns.append("REST API structure")
        if "components/" in path_str:
            patterns.append("Component-based architecture")
        if "services/" in path_str:
            patterns.append("Service layer")
        if "repositories/" in path_str:
            patterns.append("Repository pattern")
        if "middleware/" in path_str:
            patterns.append("Middleware pattern")
        if "hooks/" in path_str:
            patterns.append("Hooks pattern")
        if "utils/" in path_str or "helpers/" in path_str:
            patterns.append("Utility module pattern")
        if "tests/" in path_str or "test_" in path_str:
            patterns.append("Test suite")

        return patterns

    def _detect_conventions(self, files: dict[str, FileInfo]) -> list[str]:
        """Detect coding conventions."""
        conventions = []

        # Naming conventions
        paths = list(files.keys())

        # Check for snake_case vs camelCase
        snake_count = sum(1 for p in paths if '_' in p and not p.startswith('_'))
        camel_count = sum(1 for p in paths if re.search(r'[a-z][A-Z]', p))

        if snake_count > camel_count:
            conventions.append("snake_case naming")
        elif camel_count > snake_count:
            conventions.append("camelCase naming")

        # Check for type hints (Python)
        python_files = [f for f in files.values() if f.language == "python"]
        if python_files:
            # This is a simplification - would need to analyze actual content
            conventions.append("Python codebase")

        return conventions

    def _generate_summary(self, codebase_map: CodebaseMap) -> str:
        """Generate a summary of the codebase."""
        file_count = len(codebase_map.files)
        languages = {}
        total_lines = 0

        for info in codebase_map.files.values():
            languages[info.language] = languages.get(info.language, 0) + 1
            total_lines += info.line_count

        primary_lang = max(languages, key=languages.get) if languages else "unknown"

        summary_parts = [
            f"Codebase with {file_count} files, {total_lines} lines",
            f"Primary language: {primary_lang}",
        ]

        if codebase_map.entry_points:
            summary_parts.append(f"Entry points: {', '.join(codebase_map.entry_points[:3])}")

        if codebase_map.patterns:
            summary_parts.append(f"Patterns: {', '.join(codebase_map.patterns[:3])}")

        return " | ".join(summary_parts)

    def _cache_analysis(self, codebase_id: str, codebase_map: CodebaseMap):
        """Cache the analysis results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO codebases
            (id, root_path, summary, patterns, conventions, entry_points, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            codebase_id,
            codebase_map.root_path,
            codebase_map.summary,
            json.dumps(codebase_map.patterns),
            json.dumps(codebase_map.conventions),
            json.dumps(codebase_map.entry_points),
            datetime.now().isoformat(),
        ))

        # Cache file info
        for path, info in codebase_map.files.items():
            file_id = hashlib.md5(f"{codebase_id}_{path}".encode()).hexdigest()[:16]
            cursor.execute("""
                INSERT OR REPLACE INTO files
                (id, codebase_id, path, language, size_bytes, line_count, purpose,
                 imports, exports, functions, classes, dependencies, hash, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                codebase_id,
                path,
                info.language,
                info.size_bytes,
                info.line_count,
                info.purpose,
                json.dumps(info.imports),
                json.dumps(info.exports),
                json.dumps(info.functions),
                json.dumps(info.classes),
                json.dumps(info.dependencies),
                info.hash,
                datetime.now().isoformat(),
            ))

        conn.commit()
        conn.close()

    def _load_cached(self, codebase_id: str) -> CodebaseMap | None:
        """Load cached analysis if available and recent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT root_path, summary, patterns, conventions, entry_points, analyzed_at
            FROM codebases WHERE id = ?
        """, (codebase_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        # Check if cache is recent (within 1 hour)
        analyzed_at = datetime.fromisoformat(row[5])
        if (datetime.now() - analyzed_at).total_seconds() > 3600:
            conn.close()
            return None

        codebase_map = CodebaseMap(
            root_path=row[0],
            summary=row[1],
            patterns=json.loads(row[2]) if row[2] else [],
            conventions=json.loads(row[3]) if row[3] else [],
            entry_points=json.loads(row[4]) if row[4] else [],
        )

        # Load files
        cursor.execute("""
            SELECT path, language, size_bytes, line_count, purpose,
                   imports, exports, functions, classes, hash
            FROM files WHERE codebase_id = ?
        """, (codebase_id,))

        for file_row in cursor.fetchall():
            codebase_map.files[file_row[0]] = FileInfo(
                path=file_row[0],
                language=file_row[1],
                size_bytes=file_row[2],
                line_count=file_row[3],
                purpose=file_row[4],
                imports=json.loads(file_row[5]) if file_row[5] else [],
                exports=json.loads(file_row[6]) if file_row[6] else [],
                functions=json.loads(file_row[7]) if file_row[7] else [],
                classes=json.loads(file_row[8]) if file_row[8] else [],
                hash=file_row[9],
            )

        conn.close()
        return codebase_map

    def get_relevant_files(self, codebase_map: CodebaseMap, task: str) -> list[str]:
        """
        Find files relevant to a task using hybrid keyword + semantic matching.
        """
        from embeddings import get_embedding_service

        task_lower = task.lower()
        task_words = set(task_lower.split())
        relevant = []

        # Get task embedding for semantic comparison
        embedding_service = get_embedding_service()
        task_embedding = embedding_service.embed(task).embedding

        for path, info in codebase_map.files.items():
            score = 0.0

            # Keyword matching (fast, exact)
            path_lower = path.lower()
            path_words = set(re.findall(r'\w+', path_lower))
            path_overlap = len(task_words & path_words)
            score += path_overlap * 2

            symbols = set(f.lower() for f in info.functions + info.classes)
            symbol_overlap = len(task_words & symbols)
            score += symbol_overlap * 3

            if any(w in info.purpose.lower() for w in task_words):
                score += 2

            # Semantic matching (slower, but catches conceptual similarity)
            file_description = f"{path} {info.purpose} {' '.join(info.functions[:10])} {' '.join(info.classes[:5])}"
            file_embedding = embedding_service.embed(file_description).embedding
            semantic_similarity = embedding_service.similarity(task_embedding, file_embedding)

            # Semantic similarity contributes to score (scaled to match keyword scoring)
            if semantic_similarity > 0.3:  # Only count meaningful similarity
                score += semantic_similarity * 5

            if score > 0:
                relevant.append((path, score, semantic_similarity))

        # Sort by combined score
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _, _ in relevant[:10]]

    def get_context_for_task(self, root_path: str, task: str) -> str:
        """Get relevant context for a task."""
        codebase_map = self.analyze(root_path)
        relevant_files = self.get_relevant_files(codebase_map, task)

        context_parts = [
            f"## Codebase Overview\n{codebase_map.summary}",
        ]

        if codebase_map.patterns:
            context_parts.append(f"\n**Patterns:** {', '.join(codebase_map.patterns)}")

        if relevant_files:
            context_parts.append(f"\n## Relevant Files")
            for path in relevant_files[:5]:
                info = codebase_map.files.get(path)
                if info:
                    context_parts.append(
                        f"\n**{path}** ({info.purpose})\n"
                        f"- Functions: {', '.join(info.functions[:5])}\n"
                        f"- Classes: {', '.join(info.classes[:5])}"
                    )

        return "\n".join(context_parts)


# Global analyzer instance
_analyzer: CodebaseAnalyzer | None = None


def get_codebase_analyzer() -> CodebaseAnalyzer:
    """Get or create the global codebase analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodebaseAnalyzer()
    return _analyzer
