# How to write documentation

This guide explains how to write and contribute to the documentation for the PyMRITools project.
It covers the documentation structure, how the configuration works,
and how the documentation is automatically built and deployed.

## Documentation Structure

The documentation for this project is built using [MkDocs](https://www.mkdocs.org/), a fast and simple static site generator that's geared towards building project documentation. The documentation source files are written in Markdown and are stored in the `docs/` directory of the repository.

## Understanding the MkDocs Configuration

The `mkdocs.yml` file at the root of the repository is the configuration file for MkDocs. It defines:

1. **Basic information**: Site name, description, and author
2. **Repository information**: Repository name, URL, and edit URI
3. **Theme**: The visual theme for the documentation (we use ReadTheDocs)
4. **Markdown extensions**: Additional Markdown features enabled
5. **Navigation structure**: The organization of pages in the documentation
6. **Plugins**: Additional functionality for the documentation

When you want to **add** a new documentation page, you need to:

1. Create a Markdown file in the `docs/` directory or a subdirectory
2. Add the page to the `nav` section in `mkdocs.yml` if you want it to appear in the navigation menu

## Automatic Documentation Building with GitHub Actions

One of the powerful features of our documentation setup is that it's automatically built and deployed whenever changes are pushed to the `main` branch.
This is done using [GitHub Actions](https://github.com/features/actions), a continuous integration and continuous deployment (CI/CD) platform.

### How it Works

1. **Trigger**: When someone pushes changes to the `main` branch
2. **Build Environment**: GitHub Actions sets up a virtual machine with Ubuntu
3. **Setup**: Python is installed, along with MkDocs
4. **Build and Deploy**: MkDocs builds the documentation and deploys it to GitHub Pages

The entire process is defined in the `.github/workflows/mkdocs.yml` file:

### What This Means for Contributors

As a contributor, you don't need to worry about manually building or deploying the documentation.
When your changes are merged into the main branch:

1. GitHub Actions automatically detects the changes
2. It builds the documentation using MkDocs
3. It deploys the built documentation to the `gh-pages` branch
4. GitHub Pages serves the documentation from the `gh-pages` branch

This means you can focus on writing good documentation content without worrying about the technical details of deployment.

## Writing Documentation

When writing documentation:

1. **Use Markdown**: All documentation is written in Markdown format
2. **Follow the structure**: Place your files in the appropriate directories
3. **Update navigation**: Add your page to the `nav` section in `mkdocs.yml` if needed
4. **Preview locally**: You can run `mkdocs serve` locally to preview changes
5. **Commit and push**: Once you're satisfied, commit your changes and create a pull request

### Rules for Writing

There is a small set of rules that are non-negotiable:

1. Always write in English and use spell and grammar check.
   In PyCharm you can use the Grazie plugin for checking your English.
2. Start a new line after the end of each sentence.
   This makes it much easier to see differences in pull-requests.
3. Always remember, you are writing documentation for people who don't know how things work.
   Be precise and clear in your writing.
   Your future You will thank you.

### Markdown Tips

MkDocs supports standard Markdown syntax plus some extensions:

```markdown
# Heading 1
## Heading 2

*Italic text*
**Bold text**

- Bullet point
- Another bullet point

1. Numbered item
2. Another numbered item

[Link text](https://example.com)

![Image alt text](path/to/image.png)

`inline code`

!!! note
    This is an admonition box for notes
```

## Testing Your Documentation Locally

Before submitting your changes, it's a good idea to preview them locally:

1. Either use the conda environment from `python_setup/environment.yml` or install MkDocs: `pip install mkdocs`
2. Navigate to the repository root
3. Run `mkdocs serve`
4. Open your browser to `http://127.0.0.1:8000/`

This will give you a live preview of the documentation that updates as you make changes.
