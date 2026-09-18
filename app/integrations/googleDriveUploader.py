"""
Google Drive uploader for PersonaCluster.

Responsibilities:
    1. Authenticate the user with Google Drive.
    2. Create/find the PersonaCluster root folder.
    3. Create an event folder.
    4. Recursively reproduce the local output directory structure.
    5. Upload files into their corresponding Drive folders.
    6. Optionally make each person folder editable by anyone with the link.
    7. Return Drive folder IDs and URLs.
"""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


class GoogleDriveUploader:
    """
    Uploads PersonaCluster event output to Google Drive.

    The uploader mirrors the local directory structure inside
    a Google Drive event folder.

    Example:

        output/
        ├── Ahmed/
        │   ├── All Images/
        │   ├── Best Images/
        │   └── representative Image.jpg

    becomes:

        Event Results/
        └── Event Name/
            ├── Ahmed/
            │   ├── All Images/
            │   ├── Best Images/
            │   └── representative Image.jpg
    """

    # Google Drive API scope.

    # This allows the application to create/manage files it uploads
    # and work with the selected Drive location.
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

    def __init__(
        self,
        credentials_path: str,
        token_path: str,
        root_folder_id: str = "",
        root_folder_name: str = "Event Results",
        public_link: bool = False,
        public_link_role: str = "writer",
        retry_count: int = 3,
        http_timeout_seconds: int = 60,
    ) -> None:
        """
        Initialize the Google Drive uploader.

        Args:
            credentials_path:
                Path to Google's OAuth client credentials JSON.

            token_path:
                Path where the authorized user token will be stored.

            root_folder_id:
                Existing Google Drive folder ID to use as the root.

                If empty, root_folder_name is searched for in My Drive
                and created if necessary.

            root_folder_name:
                Name of the root folder created when no root_folder_id
                is provided.

            public_link:
                If True, EACH PERSON FOLDER receives an
                "anyone with the link" editor permission.
                The event/root folder is NOT made public.

            public_link_role:
                Drive permission role used for the person-folder link.
                For the requested behavior this should be "writer".

            retry_count:
                Number of retries for transient Drive API failures.

            http_timeout_seconds:
                HTTP timeout used by the Drive API client.
        """

        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)

        self.root_folder_id = root_folder_id.strip()
        self.root_folder_name = root_folder_name

        self.public_link = public_link
        self.public_link_role = public_link_role.strip().lower()

        if self.public_link_role not in {"reader", "commenter", "writer"}:
            raise ValueError(
                "public_link_role must be one of: reader, commenter, writer"
            )

        self.retry_count = max(0, retry_count)
        self.http_timeout_seconds = http_timeout_seconds

        self.service = None

    # ========================================================
    # PUBLIC API
    # ========================================================

    def upload_event(
        self,
        output_directory: str | Path,
        event_name: Optional[str] = None,
    ) -> dict:
        """Upload person folders and return their Drive folder links."""
        if not output_directory:
            raise ValueError("Google Drive upload requires an output directory.")

        output_directory = Path(output_directory)
        if not output_directory.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_directory}"
            )
        if not output_directory.is_dir():
            raise NotADirectoryError(
                f"Output path is not a directory: {output_directory}"
            )

        if event_name is None:
            event_name = output_directory.name
        if not event_name.strip():
            raise ValueError("Google Drive event folder name cannot be empty.")

        self._authenticate()
        root_folder_id = self._get_root_folder_id()
        event_folder_id = self._create_folder(
            name=event_name,
            parent_id=root_folder_id,
        )

        print(f"[DRIVE] Created event folder: {event_name} ({event_folder_id})")

        uploaded_files = 0
        person_folder_urls: dict[str, str] = {}

        # Upload only person folders. The final Excel is generated afterward.
        for person_directory in sorted(
            (path for path in output_directory.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        ):
            person_folder_id = self._create_folder(
                name=person_directory.name,
                parent_id=event_folder_id,
            )

            uploaded_files += self._upload_directory(
                local_directory=person_directory,
                drive_parent_id=person_folder_id,
            )

            if self.public_link:
                self._make_public(person_folder_id)

            person_folder_urls[person_directory.name] = (
                f"https://drive.google.com/drive/folders/{person_folder_id}"
            )

            print(
                f"[DRIVE] {person_directory.name}: "
                f"{person_folder_urls[person_directory.name]}"
            )

        event_folder_url = f"https://drive.google.com/drive/folders/{event_folder_id}"

        print(f"[DRIVE] Upload completed: {uploaded_files} files")
        print(f"[DRIVE] Event folder: {event_folder_url}")

        return {
            "root_folder_id": root_folder_id,
            "event_folder_id": event_folder_id,
            "event_folder_url": event_folder_url,
            "person_folder_urls": person_folder_urls,
            "uploaded_files": uploaded_files,
        }

    def upload_file_to_folder(
        self,
        local_path: str | Path,
        drive_parent_id: str,
    ) -> str:
        """Upload one file to an existing Drive folder and return its file ID."""
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"File does not exist: {local_path}")

        return self._upload_file(
            local_path=local_path,
            drive_parent_id=drive_parent_id,
        )

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    def _authenticate(self) -> None:
        """
        Authenticate the user with Google Drive.

        On the first run:
            - Opens the Google OAuth browser flow.
            - User signs into Google.
            - User grants Drive access.
            - token.json is created.

        On subsequent runs:
            - Existing token.json is reused.
            - Expired tokens are refreshed automatically.
        """

        creds = None

        # ----------------------------------------------------
        # Load previously saved credentials
        # ----------------------------------------------------

        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(self.token_path),
                    self.SCOPES,
                )
            except Exception as exc:
                print(f"[DRIVE] Could not load existing token: {exc}")

        # ----------------------------------------------------
        # Refresh or create credentials
        # ----------------------------------------------------

        if not creds or not creds.valid:

            if creds and creds.expired and creds.refresh_token:
                print("[DRIVE] Refreshing Google authorization...")

                creds.refresh(Request())

            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        "Google Drive OAuth credentials were not found.\n"
                        f"Expected: {self.credentials_path}\n\n"
                        "Download the OAuth Desktop App credentials "
                        "from Google Cloud and place them at this path."
                    )

                print("[DRIVE] Google authorization required.")

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path),
                    self.SCOPES,
                )

                creds = flow.run_local_server(port=0)

            # ------------------------------------------------
            # Save credentials
            # ------------------------------------------------

            self.token_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.token_path.open(
                "w",
                encoding="utf-8",
            ) as token_file:
                token_file.write(creds.to_json())

            print(f"[DRIVE] Authorization saved to " f"{self.token_path}")

        # ----------------------------------------------------
        # Build Drive API client
        # ----------------------------------------------------

        self.service = build(
            "drive",
            "v3",
            credentials=creds,
            cache_discovery=False,
        )

        print("[DRIVE] Google Drive authentication successful.")

    # ========================================================
    # ROOT FOLDER
    # ========================================================

    def _get_root_folder_id(self) -> str:
        """
        Return the configured root Drive folder.

        Priority:

            1. Explicit GOOGLE_DRIVE_ROOT_FOLDER_ID
            2. Existing folder with configured name
            3. Create new root folder
        """

        # ----------------------------------------------------
        # Explicit folder ID
        # ----------------------------------------------------

        if self.root_folder_id:
            print("[DRIVE] Using configured root folder ID: " f"{self.root_folder_id}")

            return self.root_folder_id

        # ----------------------------------------------------
        # Search for existing folder
        # ----------------------------------------------------

        existing_folder_id = self._find_folder(
            name=self.root_folder_name,
            parent_id=None,
        )

        if existing_folder_id:
            print("[DRIVE] Using existing root folder: " f"{self.root_folder_name}")

            return existing_folder_id

        # ----------------------------------------------------
        # Create root folder
        # ----------------------------------------------------

        root_folder_id = self._create_folder(
            name=self.root_folder_name,
            parent_id=None,
        )

        print("[DRIVE] Created root folder: " f"{self.root_folder_name}")

        return root_folder_id

    # ========================================================
    # FOLDER OPERATIONS
    # ========================================================

    def _create_folder(
        self,
        name: str,
        parent_id: Optional[str],
    ) -> str:
        """
        Create a Google Drive folder and return its ID.
        """

        metadata = {
            "name": name,
            "mimeType": self.FOLDER_MIME_TYPE,
        }

        if parent_id:
            metadata["parents"] = [parent_id]

        response = self._execute_with_retry(
            lambda: (
                self.service.files()
                .create(
                    body=metadata,
                    fields="id,name",
                )
                .execute()
            )
        )

        return response["id"]

    def _find_folder(
        self,
        name: str,
        parent_id: Optional[str],
    ) -> Optional[str]:
        """
        Find a folder by name.

        If parent_id is provided, only folders directly inside
        that parent are searched.
        """

        escaped_name = name.replace(
            "\\",
            "\\\\",
        ).replace(
            "'",
            "\\'",
        )

        query_parts = [
            f"name = '{escaped_name}'",
            f"mimeType = '{self.FOLDER_MIME_TYPE}'",
            "trashed = false",
        ]

        if parent_id:
            query_parts.append(f"'{parent_id}' in parents")
        else:
            query_parts.append("'root' in parents")

        query = " and ".join(query_parts)

        response = self._execute_with_retry(
            lambda: (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="files(id,name)",
                    pageSize=10,
                )
                .execute()
            )
        )

        folders = response.get("files", [])

        if not folders:
            return None

        return folders[0]["id"]

    # ========================================================
    # DIRECTORY UPLOAD
    # ========================================================

    def _upload_directory(
        self,
        local_directory: Path,
        drive_parent_id: str,
    ) -> int:
        """
        Recursively upload a local directory.

        Every local directory becomes a Google Drive folder.
        Every local file becomes a Google Drive file.

        Returns:
            Number of uploaded files.
        """

        uploaded_files = 0

        # Sort for deterministic processing.
        children = sorted(
            local_directory.iterdir(),
            key=lambda path: (
                not path.is_dir(),
                path.name.lower(),
            ),
        )

        for local_path in children:

            # ------------------------------------------------
            # Directory
            # ------------------------------------------------

            if local_path.is_dir():

                print(f"[DRIVE] Creating folder: " f"{local_path.name}")

                drive_folder_id = self._create_folder(
                    name=local_path.name,
                    parent_id=drive_parent_id,
                )

                uploaded_files += self._upload_directory(
                    local_directory=local_path,
                    drive_parent_id=drive_folder_id,
                )

                continue

            # ------------------------------------------------
            # File
            # ------------------------------------------------

            if local_path.is_file():

                self._upload_file(
                    local_path=local_path,
                    drive_parent_id=drive_parent_id,
                )

                uploaded_files += 1

        return uploaded_files

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    def _upload_file(
        self,
        local_path: Path,
        drive_parent_id: str,
    ) -> str:
        """
        Upload one local file to Google Drive.

        Returns:
            Google Drive file ID.
        """

        mime_type, _ = mimetypes.guess_type(str(local_path))

        if mime_type is None:
            mime_type = "application/octet-stream"

        metadata = {
            "name": local_path.name,
            "parents": [drive_parent_id],
        }

        media = MediaFileUpload(
            str(local_path),
            mimetype=mime_type,
            resumable=True,
        )

        print(f"[DRIVE] Uploading: " f"{local_path.name}")

        response = self._execute_with_retry(
            lambda: (
                self.service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id,name,webViewLink",
                )
                .execute()
            )
        )

        return response["id"]

    # ========================================================
    # SHARING
    # ========================================================

    def _make_public(
        self,
        folder_id: str,
    ) -> None:
        """
        Make the event folder accessible to anyone who has
        the link with read-only permission.

        Child files/folders inherit permissions from the parent
        folder.
        """

        permission = {
            "type": "anyone",
            "role": self.public_link_role,
        }

        print("[DRIVE] Making person folder editable by anyone with the link...")

        self._execute_with_retry(
            lambda: (
                self.service.permissions()
                .create(
                    fileId=folder_id,
                    body=permission,
                    fields="id,type,role",
                )
                .execute()
            )
        )

        print(f"[DRIVE] Anyone-with-link {self.public_link_role} permission created.")

    # ========================================================
    # RETRY / ERROR HANDLING
    # ========================================================

    def _execute_with_retry(
        self,
        operation,
    ):
        """
        Execute a Google Drive API operation with retries.

        Retries are primarily intended for transient API/network
        failures.
        """

        last_error = None

        attempts = self.retry_count + 1

        for attempt in range(1, attempts + 1):

            try:
                return operation()

            except HttpError as exc:

                last_error = exc

                status = getattr(
                    exc.resp,
                    "status",
                    None,
                )

                # Retry common transient statuses.
                retryable = status in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }

                if not retryable:
                    raise

                if attempt >= attempts:
                    raise

                delay = min(
                    2 ** (attempt - 1),
                    10,
                )

                print(
                    f"[DRIVE] Temporary API error "
                    f"{status}. "
                    f"Retrying in {delay}s..."
                )

                time.sleep(delay)

            except Exception as exc:

                last_error = exc

                if attempt >= attempts:
                    raise

                delay = min(
                    2 ** (attempt - 1),
                    10,
                )

                print("[DRIVE] Temporary error: " f"{exc}. " f"Retrying in {delay}s...")

                time.sleep(delay)

        if last_error:
            raise last_error

        raise RuntimeError("Google Drive operation failed without an error.")
