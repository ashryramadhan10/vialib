import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="<ashry-ramadhan>", # Replace with your username

    version="1.0.0",

    author="<ashryramadhan>",

    author_email="<ashry.ramadhan10@gmail.com>",

    description="<Template Setup.py package>",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="<https://github.com/ashryramadhan10/vialib>",

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.8',

)
