import os


def pytest_configure():
    os.environ["TATARA_API_KEY"] = "fake_test_api_key"
