"""
GCP storage client for handling file operations with Google Cloud Storage.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, BinaryIO
from dataclasses import dataclass

from google.cloud import storage
from google.oauth2 import service_account
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_config, GCPConfig

logger = logging.getLogger(__name__)


@dataclass
class GCSFileInfo:
    """Information about a file in GCS."""
    name: str
    bucket: str
    size: int
    updated: datetime
    gcs_path: str


class GCPStorageClient:
    """Client for interacting with Google Cloud Storage."""

    def __init__(self, config: Optional[GCPConfig] = None):
        """Initialize the GCP storage client.

        Args:
            config: GCP configuration. If not provided, uses global config.
        """
        self.config = config or get_config().gcp
        self._client: Optional[storage.Client] = None

    def _get_client(self) -> storage.Client:
        """Get or create the GCS client."""
        if self._client is None:
            credentials = None
            if self.config.credentials_path and os.path.exists(self.config.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                credentials = service_account.Credentials.from_service_account_file(
                    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                )

            self._client = storage.Client(
                project=self.config.project_id,
                credentials=credentials
            )
        return self._client

    def _get_bucket(self, bucket_name: str) -> storage.Bucket:
        """Get a bucket by name.

        Args:
            bucket_name: Name of the bucket.

        Returns:
            The bucket object.
        """
        client = self._get_client()
        return client.bucket(bucket_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def download_file(
        self,
        source_gcs_path: str,
        dest_path: str,
        bucket_name: Optional[str] = None
    ) -> str:
        """Download a file from GCS to local path.

        Args:
            source_gcs_path: GCS path (e.g., "bucket/path/to/file.h5")
            dest_path: Local destination path
            bucket_name: Bucket name. Uses config source_bucket if not provided.

        Returns:
            The local file path.
        """
        bucket_name = bucket_name or self.config.source_bucket
        bucket = self._get_bucket(bucket_name)

        # Parse the GCS path
        path_parts = source_gcs_path.split("/")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GCS path: {source_gcs_path}")

        blob_name = "/".join(path_parts[1:]) if path_parts[0] == bucket_name else source_gcs_path

        # Ensure destination directory exists
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        blob = bucket.blob(blob_name)
        logger.info(f"Downloading gs://{bucket_name}/{blob_name} to {dest_path}")

        blob.download_to_filename(dest_path)
        logger.info(f"Successfully downloaded {source_gcs_path}")

        return dest_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def upload_file(
        self,
        source_path: str,
        dest_gcs_path: str,
        bucket_name: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """Upload a local file to GCS.

        Args:
            source_path: Local file path
            dest_gcs_path: GCS destination path (e.g., "bucket/path/to/file.h5")
            bucket_name: Bucket name. Uses config dest_bucket if not provided.
            content_type: MIME type of the file

        Returns:
            The GCS path.
        """
        bucket_name = bucket_name or self.config.dest_bucket
        bucket = self._get_bucket(bucket_name)

        # Parse the GCS path
        path_parts = dest_gcs_path.split("/")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GCS path: {dest_gcs_path}")

        blob_name = "/".join(path_parts[1:]) if path_parts[0] == bucket_name else dest_gcs_path

        blob = bucket.blob(blob_name)

        logger.info(f"Uploading {source_path} to gs://{bucket_name}/{blob_name}")

        if content_type:
            blob.upload_from_filename(source_path, content_type=content_type)
        else:
            blob.upload_from_filename(source_path)

        logger.info(f"Successfully uploaded to gs://{bucket_name}/{blob_name}")

        return dest_gcs_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def list_files(
        self,
        prefix: str = "",
        bucket_name: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 100
    ) -> List[GCSFileInfo]:
        """List files in a GCS prefix.

        Args:
            prefix: Prefix to filter files
            bucket_name: Bucket name. Uses config source_bucket if not provided.
            extension: Filter by file extension (e.g., ".h5")
            max_results: Maximum number of results to return

        Returns:
            List of GCSFileInfo objects.
        """
        bucket_name = bucket_name or self.config.source_bucket
        bucket = self._get_bucket(bucket_name)

        prefix = f"{self.config.source_prefix}/{prefix}".lstrip("/")

        logger.info(f"Listing files in gs://{bucket_name}/{prefix}")

        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)

        files = []
        for blob in blobs:
            if extension and not blob.name.endswith(extension):
                continue

            files.append(GCSFileInfo(
                name=os.path.basename(blob.name),
                bucket=bucket_name,
                size=blob.size,
                updated=blob.updated,
                gcs_path=f"{bucket_name}/{blob.name}"
            ))

        logger.info(f"Found {len(files)} files")

        return files

    def get_latest_file(
        self,
        pattern: str = "",
        bucket_name: Optional[str] = None
    ) -> Optional[GCSFileInfo]:
        """Get the latest file matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "2026-01-12_*.h5")
            bucket_name: Bucket name. Uses config source_bucket if not provided.

        Returns:
            The latest GCSFileInfo or None if no files found.
        """
        files = self.list_files(prefix=pattern, bucket_name=bucket_name)

        if not files:
            return None

        # Sort by updated time, most recent first
        files.sort(key=lambda f: f.updated, reverse=True)

        return files[0]

    def get_file_by_pattern(
        self,
        pattern: str,
        bucket_name: Optional[str] = None
    ) -> List[GCSFileInfo]:
        """Get files matching a pattern, sorted by most recent first.

        Args:
            pattern: Glob-style pattern (e.g., "2026-01-12_*.h5")
            bucket_name: Bucket name. Uses config source_bucket if not provided.

        Returns:
            List of matching GCSFileInfo objects, sorted by most recent first.
        """
        import fnmatch

        files = self.list_files(bucket_name=bucket_name)

        matching = [
            f for f in files
            if fnmatch.fnmatch(f.name, pattern)
        ]

        # Sort by updated time, most recent first
        matching.sort(key=lambda f: f.updated, reverse=True)

        return matching

    def file_exists(
        self,
        gcs_path: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """Check if a file exists in GCS.

        Args:
            gcs_path: GCS path
            bucket_name: Bucket name

        Returns:
            True if file exists.
        """
        bucket_name = bucket_name or self.config.source_bucket

        try:
            path_parts = gcs_path.split("/")
            blob_name = "/".join(path_parts[1:]) if path_parts[0] == bucket_name else gcs_path

            bucket = self._get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

    def delete_file(
        self,
        gcs_path: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """Delete a file from GCS.

        Args:
            gcs_path: GCS path
            bucket_name: Bucket name

        Returns:
            True if deletion was successful.
        """
        bucket_name = bucket_name or self.config.source_bucket

        try:
            path_parts = gcs_path.split("/")
            blob_name = "/".join(path_parts[1:]) if path_parts[0] == bucket_name else gcs_path

            bucket = self._get_bucket(bucket_name)
            blob = bucket.blob(blob_name)

            blob.delete()
            logger.info(f"Deleted gs://{gcs_path}")

            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False

    def close(self) -> None:
        """Close the GCS client."""
        if self._client:
            self._client.close()
            self._client = None


# Global client instance
_gcs_client: Optional[GCPStorageClient] = None


def get_gcs_client() -> GCPStorageClient:
    """Get the global GCS client instance."""
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = GCPStorageClient()
    return _gcs_client


def reset_gcs_client() -> None:
    """Reset the global GCS client instance."""
    if _gcs_client:
        _gcs_client.close()
    _gcs_client = None