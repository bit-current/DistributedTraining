from setuptools import setup, find_packages

setup(
    name='hivetrain',
    version='0.2.7',
    author='Hivetrain',
    author_email='test@test.com',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourgithubusername/your_project_repo',
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
