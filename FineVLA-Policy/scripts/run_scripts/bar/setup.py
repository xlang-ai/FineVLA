from setuptools import setup, find_packages

setup(
    name="starVLA",
    version="1.0.1",
    author="Jinhui Ye, Fangjing Wang, Junqiu Yu",
    author_email="jinhuiyes@gmail.com",
    description="StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # ????????????????????????
    ],
)
