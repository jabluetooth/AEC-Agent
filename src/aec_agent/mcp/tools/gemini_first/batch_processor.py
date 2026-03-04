"""
Batch PDF Processing Module.

Enables processing multiple PDF files in parallel with progress tracking,
error isolation per file, and comprehensive reporting.

Features:
- Process directories of PDFs
- Parallel processing with asyncio (configurable concurrency)
- Progress reporting
- Per-file error isolation (one failure doesn't stop the batch)
- Summary report generation
- Resume capability for failed files
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class BatchStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some files succeeded, some failed


class FileStatus(str, Enum):
    """Status of a single file in a batch."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileResult:
    """Result of processing a single file."""
    file_path: str
    status: FileStatus
    duration_ms: float = 0.0
    entity_count: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "entity_count": self.entity_count,
            "output_path": self.output_path,
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_dir: Path
    output_dir: Path
    file_pattern: str = "*.pdf"
    max_parallel: int = 4  # Maximum concurrent files
    continue_on_error: bool = True  # Continue if a file fails
    create_output_dirs: bool = True  # Create output directories
    overwrite_existing: bool = False  # Overwrite existing output files
    export_dxf: bool = True  # Export to DXF in addition to/instead of AutoCAD
    create_in_autocad: bool = False  # Create entities in AutoCAD

    # Pipeline configuration
    extraction_method: str = "hybrid"
    preprocess: bool = True
    use_symbol_rag: bool = True
    refine_geometry: bool = True
    validate: bool = False
    dpi: int = 300

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class BatchResult:
    """Result of a batch processing job."""
    batch_id: UUID
    status: BatchStatus
    total_files: int
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    duration_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    file_results: List[FileResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.success_count / self.total_files

    def to_dict(self) -> dict:
        return {
            "batch_id": str(self.batch_id),
            "status": self.status.value,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
            "success_rate": f"{self.success_rate:.1%}",
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "file_results": [r.to_dict() for r in self.file_results],
            "errors": self.errors,
        }

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Batch ID: {self.batch_id}",
            f"Status: {self.status.value}",
            f"Total Files: {self.total_files}",
            f"Success: {self.success_count} ({self.success_rate:.1%})",
            f"Failed: {self.failure_count}",
            f"Skipped: {self.skipped_count}",
            f"Duration: {self.duration_seconds:.1f}s",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


class BatchProcessor:
    """
    Processes multiple PDF files in parallel.

    Example:
        >>> from aec_agent.mcp.tools.gemini_first import BatchProcessor, BatchConfig
        >>> config = BatchConfig(
        ...     input_dir="/pdfs",
        ...     output_dir="/output",
        ...     max_parallel=4,
        ... )
        >>> processor = BatchProcessor(config)
        >>> result = await processor.process()
        >>> print(f"Processed {result.success_count} of {result.total_files} files")
    """

    def __init__(
        self,
        config: BatchConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Initialize batch processor.

        Args:
            config: Batch configuration
            progress_callback: Optional callback(current, total, filename) for progress
        """
        self.config = config
        self.progress_callback = progress_callback
        self.batch_id = uuid4()
        self._cancelled = False
        self._semaphore: Optional[asyncio.Semaphore] = None

    def cancel(self) -> None:
        """Cancel the batch processing."""
        self._cancelled = True
        logger.info("batch_cancelled", batch_id=str(self.batch_id))

    def _find_files(self) -> List[Path]:
        """Find all files matching the pattern."""
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")

        files = list(self.config.input_dir.glob(self.config.file_pattern))

        # Also search subdirectories
        files.extend(self.config.input_dir.glob(f"**/{self.config.file_pattern}"))

        # Remove duplicates and sort
        files = sorted(set(files))

        logger.info(
            "batch_files_found",
            count=len(files),
            pattern=self.config.file_pattern,
            input_dir=str(self.config.input_dir),
        )

        return files

    def _get_output_path(self, input_file: Path, extension: str = ".dxf") -> Path:
        """Get output path for a file."""
        # Preserve subdirectory structure
        relative = input_file.relative_to(self.config.input_dir)
        output_path = self.config.output_dir / relative.with_suffix(extension)

        return output_path

    async def _process_single_file(
        self,
        file_path: Path,
        file_index: int,
        total_files: int,
    ) -> FileResult:
        """Process a single file."""
        import time

        start_time = time.perf_counter()

        result = FileResult(
            file_path=str(file_path),
            status=FileStatus.PENDING,
        )

        if self._cancelled:
            result.status = FileStatus.SKIPPED
            return result

        # Check if output exists
        output_path = self._get_output_path(file_path, ".dxf")
        if output_path.exists() and not self.config.overwrite_existing:
            result.status = FileStatus.SKIPPED
            result.output_path = str(output_path)
            result.warnings.append("Output file exists, skipped")
            return result

        result.status = FileStatus.PROCESSING

        # Report progress
        if self.progress_callback:
            self.progress_callback(file_index + 1, total_files, file_path.name)

        try:
            async with self._semaphore:
                # Import here to avoid circular imports
                from .unified_pipeline import (
                    UnifiedPipeline,
                    PipelineConfig,
                    ExtractionMethod,
                    SymbolMethod,
                )

                # Create pipeline config
                pipeline_config = PipelineConfig(
                    dpi=self.config.dpi,
                    preprocess=self.config.preprocess,
                    extraction_method=ExtractionMethod(self.config.extraction_method),
                    symbol_method=SymbolMethod.RAG if self.config.use_symbol_rag else SymbolMethod.HARDCODED,
                    refine_geometry=self.config.refine_geometry,
                    create_in_autocad=self.config.create_in_autocad,
                    validate=self.config.validate,
                )

                # Run pipeline
                pipeline = UnifiedPipeline(pipeline_config)
                pipeline_result = await pipeline.process(
                    pdf_path=str(file_path),
                    page=1,
                    output_dir=str(output_path.parent),
                )

                if not pipeline_result.success:
                    raise Exception(pipeline_result.error or "Pipeline failed")

                result.entity_count = pipeline_result.total_entities

                # Export to DXF if requested
                if self.config.export_dxf and pipeline_result.entities:
                    from .dxf_export import export_to_dxf, is_ezdxf_available

                    if is_ezdxf_available():
                        # Create output directory
                        if self.config.create_output_dirs:
                            output_path.parent.mkdir(parents=True, exist_ok=True)

                        dxf_result = export_to_dxf(
                            pipeline_result.entities,
                            output_path,
                        )

                        if dxf_result.success:
                            result.output_path = dxf_result.output_path
                        else:
                            result.warnings.extend(dxf_result.errors)
                    else:
                        result.warnings.append("ezdxf not installed, DXF export skipped")

                result.status = FileStatus.COMPLETED

        except Exception as e:
            result.status = FileStatus.FAILED
            result.error = str(e)
            logger.warning(
                "batch_file_failed",
                file=str(file_path),
                error=str(e),
            )

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def process(self) -> BatchResult:
        """
        Process all files in the batch.

        Returns:
            BatchResult with status and per-file results
        """
        import time

        result = BatchResult(
            batch_id=self.batch_id,
            status=BatchStatus.RUNNING,
            total_files=0,
            start_time=datetime.now(),
        )

        start_time = time.perf_counter()

        try:
            # Find files
            files = self._find_files()
            result.total_files = len(files)

            if result.total_files == 0:
                result.status = BatchStatus.COMPLETED
                result.warnings = ["No files found matching pattern"]
                return result

            # Create semaphore for concurrency control
            self._semaphore = asyncio.Semaphore(self.config.max_parallel)

            logger.info(
                "batch_started",
                batch_id=str(self.batch_id),
                total_files=result.total_files,
                max_parallel=self.config.max_parallel,
            )

            # Process files
            tasks = []
            for i, file_path in enumerate(files):
                task = self._process_single_file(file_path, i, result.total_files)
                tasks.append(task)

            # Run all tasks
            file_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for file_result in file_results:
                if isinstance(file_result, Exception):
                    result.failure_count += 1
                    result.errors.append(str(file_result))
                elif isinstance(file_result, FileResult):
                    result.file_results.append(file_result)
                    if file_result.status == FileStatus.COMPLETED:
                        result.success_count += 1
                    elif file_result.status == FileStatus.FAILED:
                        result.failure_count += 1
                    elif file_result.status == FileStatus.SKIPPED:
                        result.skipped_count += 1

            # Determine final status
            if self._cancelled:
                result.status = BatchStatus.CANCELLED
            elif result.failure_count == 0:
                result.status = BatchStatus.COMPLETED
            elif result.success_count == 0:
                result.status = BatchStatus.FAILED
            else:
                result.status = BatchStatus.PARTIAL

        except Exception as e:
            result.status = BatchStatus.FAILED
            result.errors.append(str(e))
            logger.exception("batch_failed", error=str(e))

        finally:
            result.end_time = datetime.now()
            result.duration_seconds = time.perf_counter() - start_time

        logger.info(
            "batch_completed",
            batch_id=str(self.batch_id),
            status=result.status.value,
            success=result.success_count,
            failed=result.failure_count,
            duration=f"{result.duration_seconds:.1f}s",
        )

        return result


async def process_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    file_pattern: str = "*.pdf",
    max_parallel: int = 4,
    extraction_method: str = "hybrid",
    export_dxf: bool = True,
    create_in_autocad: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchResult:
    """
    Process a batch of PDF files.

    Convenience function for simple batch processing.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        file_pattern: Glob pattern for files (default "*.pdf")
        max_parallel: Maximum concurrent files
        extraction_method: Extraction method (direct, hybrid, best, vtracer)
        export_dxf: Export to DXF files
        create_in_autocad: Create entities in AutoCAD
        progress_callback: Optional progress callback(current, total, filename)

    Returns:
        BatchResult with status and per-file results

    Example:
        >>> result = await process_batch(
        ...     input_dir="/pdfs",
        ...     output_dir="/output",
        ...     max_parallel=4,
        ... )
        >>> print(result.to_summary())
    """
    config = BatchConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        file_pattern=file_pattern,
        max_parallel=max_parallel,
        extraction_method=extraction_method,
        export_dxf=export_dxf,
        create_in_autocad=create_in_autocad,
    )

    processor = BatchProcessor(config, progress_callback)
    return await processor.process()


def process_batch_sync(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    **kwargs,
) -> BatchResult:
    """
    Synchronous wrapper for process_batch.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        **kwargs: Additional arguments passed to process_batch

    Returns:
        BatchResult
    """
    return asyncio.run(process_batch(input_dir, output_dir, **kwargs))


# Batch job registry for status tracking
_active_batches: Dict[str, BatchProcessor] = {}


def get_batch_status(batch_id: str) -> Optional[dict]:
    """Get status of a batch job."""
    if batch_id in _active_batches:
        processor = _active_batches[batch_id]
        return {
            "batch_id": batch_id,
            "cancelled": processor._cancelled,
        }
    return None


def cancel_batch(batch_id: str) -> bool:
    """Cancel a running batch job."""
    if batch_id in _active_batches:
        _active_batches[batch_id].cancel()
        return True
    return False
