#!/usr/bin/env python
"""
HuggingFace Hub Token and Repository Access Checker.

This module provides utilities to verify HuggingFace authentication tokens and check
repository access permissions. It can be run as a standalone script or imported as
part of the wet-net CLI toolset.

The tool performs the following checks:
- Validates HuggingFace token from environment variables
- Retrieves authenticated user information
- Checks read/write access to specified repositories
- Displays organization roles and permissions
- Provides clear feedback on access levels and potential issues

Typical usage:
    $ wet-net hf-check WetNet/wet-net
    $ wet-net hf-check WetNet/wet-net --env-var MY_HF_TOKEN
"""

import argparse
import os
import sys

import typer
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

app = typer.Typer()


def find_token(env_var: str | None):
    """
    Locate a HuggingFace authentication token from environment variables.

    Searches for a valid HuggingFace token in environment variables. If a specific
    env_var is provided, only that variable is checked. Otherwise, it tries common
    default names: HF_TOKEN and HUGGINGFACE_HUB_TOKEN.

    Args:
        env_var: Specific environment variable name to check. If None, tries common names.

    Returns:
        tuple: A (token, env_name) tuple where:
            - token (str | None): The token value if found, None otherwise
            - env_name (str | None): The environment variable name where token was found,
              None if no token was found
    """
    if env_var:
        token = os.environ.get(env_var)
        return token, env_var if token is not None else (None, None)

    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value, name

    return None, None


@app.command()
def hf_check(
    repo_id: str = typer.Argument(..., help="Hugging Face repo id, e.g. 'WetNet/wet-net'"),
    env_var: str | None = typer.Option(
        None,
        "--env-var",
        help="Name of the env var holding the HF token (default: tries HF_TOKEN, then HUGGINGFACE_HUB_TOKEN).",
    ),
):
    """
    Check HuggingFace token validity and repository access permissions.

    This command verifies that your HuggingFace authentication token is valid and
    checks your access level to a specified repository. It displays:
    - Token authentication status
    - User and organization information
    - Repository visibility (public/private)
    - Inferred read/write permissions

    Example usage:
        wet-net hf-check WetNet/wet-net
        wet-net hf-check WetNet/wet-net --env-var MY_CUSTOM_TOKEN
    """
    token, env_name = find_token(env_var)

    # Validate that a token was found; exit early if not
    if token is None:
        checked = [env_var] if env_var else ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
        checked_names = ", ".join(n for n in checked if n)
        typer.secho(f"❌ No HF token found. Checked env vars: {checked_names}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Initialize HuggingFace API client with the located token
    api = HfApi(token=token)

    typer.secho(f"🔑 Using token from env var: {env_name}", fg=typer.colors.GREEN)
    typer.echo(f"   Token preview: {token[:6]}... (hidden for safety)")

    # --- Retrieve authenticated user information ---
    try:
        who = api.whoami(token=token)
    except Exception as e:
        typer.secho(f"⚠️ Could not retrieve token/user info: {e}", fg=typer.colors.YELLOW)
        who = None

    username = None
    org_role_for_owner = None
    repo_owner = repo_id.split("/")[0]  # Extract owner from repo_id (e.g., 'WetNet' from 'WetNet/wet-net')

    if who:
        username = who.get("name") or who.get("username")
        typer.secho(f"\n👤 Authenticated as HF user: {username}", fg=typer.colors.CYAN)

        # Extract token scopes and roles (API v2 response structure)
        auth = who.get("auth") or {}
        access_token = auth.get("accessToken") or {}
        scopes = access_token.get("scopes")
        role = access_token.get("role")

        if scopes:
            typer.echo("   Token scopes: " + ", ".join(scopes))
        if role:
            typer.echo(f"   Token role: {role}")

        # Display organization memberships and track role for target repo owner
        orgs = who.get("orgs") or []
        if orgs:
            typer.secho("\n🏢 Organization roles from token:", fg=typer.colors.CYAN)
            for org in orgs:
                typer.echo(f"   - {org.get('name')}: {org.get('role')}")
                # Track if user has a role in the organization that owns the target repo
                if org.get("name") == repo_owner:
                    org_role_for_owner = org.get("role")

    # --- Verify repository accessibility ---
    typer.secho(f"\n📦 Checking access to repo: {repo_id}", fg=typer.colors.CYAN)
    try:
        repo_info = api.repo_info(repo_id)
        is_private = getattr(repo_info, "private", None)
        typer.secho("   ✅ Read access: OK", fg=typer.colors.GREEN)
        if is_private is not None:
            visibility = "private" if is_private else "public"
            typer.echo(f"   Repo visibility: {visibility}")
    except HfHubHTTPError as e:
        typer.secho(f"   ❌ Cannot read repo '{repo_id}': {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    # --- Infer repository permissions (best-effort analysis) ---
    # Note: Full permission details require admin access and are not always exposed via API
    typer.secho("\n🔐 Inferred permissions (best-effort):", fg=typer.colors.CYAN)
    if username and repo_owner == username:
        # User owns the repository directly
        typer.secho(
            "   • You are the *owner* of this repo → you have full read/write permissions.",
            fg=typer.colors.GREEN,
        )
    elif org_role_for_owner:
        # User is a member of the organization that owns the repository
        typer.echo(
            f"   • Repo owner '{repo_owner}' is an org where your role is '{org_role_for_owner}'. "
            "Your effective permissions follow that org role."
        )
    else:
        # Generic read access confirmed; write access unclear
        typer.echo(
            "   • You can read this repo. "
            "Write/delete permissions depend on whether you are a collaborator or org member "
            "with write access, which is not fully exposed via the public API."
        )


def main():
    """
    Main entry point for the HuggingFace token and repository checker.

    Parses command-line arguments, locates the HuggingFace token, authenticates with
    the HuggingFace Hub, and performs a comprehensive check of repository access
    permissions.

    Command-line arguments:
        repo_id (str): HuggingFace repository identifier (e.g., 'WetNet/wet-net')
        --env-var (str, optional): Name of environment variable containing the HF token.
                                   Defaults to checking HF_TOKEN and HUGGINGFACE_HUB_TOKEN.

    Exit codes:
        0: Success - token found and repository accessible
        1: Failure - token not found or repository inaccessible

    Example:
        $ python hf_util.py WetNet/wet-net
        $ python hf_util.py WetNet/wet-net --env-var MY_CUSTOM_TOKEN
    """
    parser = argparse.ArgumentParser(description="Check HF token and repo permissions.")
    parser.add_argument(
        "repo_id",
        help="Hugging Face repo id, e.g. 'WetNet/wet-net'",
    )
    parser.add_argument(
        "--env-var",
        help="Name of the env var holding the HF token (default: tries HF_TOKEN, then HUGGINGFACE_HUB_TOKEN).",
        default=None,
    )
    args = parser.parse_args()

    # Call the typer command function directly with parsed arguments
    try:
        hf_check(repo_id=args.repo_id, env_var=args.env_var)
    except typer.Exit as e:
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
