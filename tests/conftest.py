'''Here for defining whole test directory wide things

Conftest is used in pytest for things like:
    - Serve fixtures to tests (no matter the fixture scope)
    - Modularize the available fixture for each conftest.py in a Tree like hierarchy
    - Scope: sharing fixtures across classes, modules, packages or session
    - Child conftest.py can override parent conftest.py (but not the opposite) See this
    - Serve fixtures from a third-party plugins
    - Loading and Sharing test data across multiple tests
    - Using fixtures from other projects
    - Serve and share Hooks

For a more comprehensive overview: https://stackoverflow.com/questions/34466027/what-is-conftest-py-for-in-pytest
'''