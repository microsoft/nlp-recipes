

# Recipe Template


# Table of Contents
- [Goal of Template Project](#goal-of-template-project)
- [Required Folders and Files](#required-folders-and-files)
- [Strongly Recommended](#strongly-recommended)
- [Recommended Folders and Files](#recommended-folders-and-files)
- [Files to Help You Create Your Project](#files-to-help-you-create-your-project) 
- [Microsoft Style Guide](#microsoft-style-guide)
- [Future](#future)

## Goal of Template Project
The intent of the Recipe Template is to make it easier to set up subsequent Recipe projects by:
- providing common base templates for notebook and other files
- simplifying the ability to capture repo metrics consistently across projects
- providing a framework for automated testing so that you have less work to do for this effort 
- enabling easier integration with Azure DevOps CI/CD (more coming) 
- providing required Microsoft files for OSS
- providing recommendations to get your project up and running quickly by leveraging prior work  


## Required Folders and Files

Some files and folders in the Recipe Template are recommendations and some are requirements. 
These are the required folders that MUST NOT BE RENAMED OR CHANGED:


    scripts/repo_metrics - metrics for the github repo which are stored in CosmosDB
    tests - This folder name is used in automated test scripts and expects the following folders, unit, smoke and integration 
    tests/unit - unit tests are run when a PR is submitted
    tests/smoke - smoke tests are run nightly
    tests/integration - check trigger (bz)

This file is required:

    LICENSE - This required file contains the MIT license and is required by Microsoft for open source projects.  Do not edit it.

## Strongly Recommended
These files are strongly recommended to include in your repo:

    configuration.ipynb
    .gitignore
    AUTHORS.md
    CONTRIBUTING.md
    SETUP.md
    chglog.txt
    codeofconduct.md

## Recommended Folders and Files

Recommendations are provided for additional folders and files to assist in setting up a new repo. If you feel strongly about one of these names, you can change it or delete.  If you believe additional root folders are required, please add an issue to the Recipe Template github.

These folders are recommendations:

    benchmarks - some repos target different verticals with their benchmarks and create folders in benchmarks for that purpose.
    docs - Some repos have docs, some do not and store their info in the example notebooks or scripts
    examples - This stores both notebooks as well as .py or .r files
    models - Thus far, models have been used by performance related repos.
    utils_xxx (or xxx_utils) - Naming utils is your preference.

Although there are recommendations for some folders in the root, you may delete those that you do not need. Again, please do not delete those indicated in [Required Folders and Files](#required-folders-and-files).

## Files to Help You Create Your Project
 
 Additional files provide guidance on creating the README, Jupyter Notebook template and insights on AzureML.
 These are for your use and should NOT be included in your repo:
 
    ML Notebook Plan.docx - This file contains a thorough description of how to create an effective Jupyter NB and was created by Jamie.                             Some of the content from the .docx are included in this README.
    Notebook Template.ipynb - This template is based on the description in the ML Notebook Plan.docx.

Please review the ML Notebook Plan.docx for details and additional information.

## Microsoft Style Guide
The [Microsoft Style Guide](https://aka.ms/style) has been vetted over the years and contains a wealth of condensed knowledge.
We strongly recommend reading it.

## Future
In the future, additional files for DevOps will be added to require less effort to integrate with Github for unit, smoke and integration tests.  Improvements are currently underway to improve the integration of DevOps with AzureML and updates will be added as they become available.






