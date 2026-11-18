from memory.compressor import _parse_response


SAMPLE_RESPONSE = """## Title
JWT Auth Middleware — halfway through refresh token rotation

## Project
ecommerce

## What we were doing
Building JWT authentication for a React + Node.js app.

## What was completed
- Login page UI
- JWT token generation

## Key decisions made
- httpOnly cookies for XSS protection

## Current state / where we stopped
Halfway through refresh token middleware.

## Next steps
1. Finish error handling
2. Add protected routes

## Important context
- Backend: Express.js port 3001

## Tags
react, nodejs, jwt, auth"""


def test_parse_title():
    result = _parse_response(SAMPLE_RESPONSE)
    assert result.title == "JWT Auth Middleware — halfway through refresh token rotation"


def test_parse_project():
    result = _parse_response(SAMPLE_RESPONSE)
    assert result.project == "ecommerce"


def test_parse_tags():
    result = _parse_response(SAMPLE_RESPONSE)
    assert "jwt" in result.tags
    assert "auth" in result.tags
    assert "react" in result.tags


def test_parse_summary_contains_full_body():
    result = _parse_response(SAMPLE_RESPONSE)
    assert "## What was completed" in result.summary
    assert "## Next steps" in result.summary


def test_parse_missing_title_raises():
    import pytest
    with pytest.raises(ValueError, match="no ## Title section"):
        _parse_response("## Project\ndefault\n## Tags\npython")


def test_parse_missing_project_defaults():
    result = _parse_response("## Title\nSomething\n## Tags\npython")
    assert result.project == "default"


def test_parse_missing_tags_defaults_empty():
    result = _parse_response("## Title\nSomething\n## Project\nmyapp")
    assert result.tags == []


def test_parse_tags_strips_whitespace():
    result = _parse_response("## Title\nT\n## Project\np\n## Tags\n react , nodejs ,  jwt ")
    assert result.tags == ["react", "nodejs", "jwt"]
