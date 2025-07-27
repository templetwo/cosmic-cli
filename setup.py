
from setuptools import setup, find_packages

setup(
    name='cosmic-cli',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'python-dotenv',
        'xai-sdk',
        'rich',
        'pyfiglet',
    ],
    entry_points={
        'console_scripts': [
            'cosmic-cli=cosmic_cli.main:cli',
        ],
    },
    author='Flamebearer',
    author_email='your.email@example.com', # You can change this later
    description="The Ultimate Cosmic CLI: Grok's Terminal Portal - Smarter, Funnier, and More Powerful than Gemini/Claude",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_github_username/cosmic-cli', # You can change this later
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Utilities',
    ],
    keywords='cosmic xai cli ai terminal',
    extras_require={
        'test': [
            'pytest',
            'pytest-asyncio',
            'pytest-cov',
            'asynctest',
            'requests-mock'
        ]
    },
)
