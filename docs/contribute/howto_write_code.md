# Rules for Writing Code

There are a few rules when you write code that need to be followed:

1. We write unit tests wherever possible, and all tests must pass before code is merged.
2. We develop on branches and use pull requests and code reviews.
3. Variable names and code documentation must always be in English.
4. We always try to use Python type hints for your functions and methods
5. We always write docstrings for functions, classes, modules, etc.
   Use [Google style documentation](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## 1. Why We Write Unit Tests Wherever Possible

Unit tests are fundamental to maintaining code quality and preventing regression bugs. 
They serve multiple critical purposes:

- They verify that individual components work as expected
- They act as documentation by showing how code should be used
- They make it safer to refactor code by quickly catching unintended changes
- They help catch bugs early in the development process
- They reduce the cost of fixing bugs by identifying issues before they reach production

## 2. Why We Use Branches and Pull Requests

Working with branches and conducting code reviews is essential for maintaining code quality and team collaboration:

- Branches allow developers to work on features or fixes without affecting the main codebase
- Pull requests provide a formal process for code integration
- Code reviews help catch bugs and design issues early
- Team members can learn from each other's code and share knowledge
- It maintains a clean and stable main branch
- It creates documentation of why changes were made through PR descriptions and review comments

## 3. Why We Use English for Variables and Documentation

Using English throughout the codebase ensures:

- Global accessibility and understanding of the code
- Consistency across the entire codebase
- Easier collaboration with international team members
- Better integration with most programming languages and libraries, which use English keywords
- Simpler maintenance and support as English is the de facto standard in programming

## 4. Why We Use Type Hints

Type hints improve code quality and development experience by:

- Making code more self-documenting
- Enabling better IDE support with accurate code completion and refactoring
- Catching type-related bugs before runtime
- Making it easier for new developers to understand function interfaces
- Improving maintainability by clarifying code intentions
- Supporting static type checking tools like mypy

## 5.  Why We Write Docstrings

Docstrings using Google style are crucial because they:

- Are automatically included in the documentation website
- Provide clear and standardized documentation for code components
- Help developers understand how to use functions, classes, and modules
- Enable automatic documentation generation
- Make code more maintainable by explaining complex logic or algorithms
- Support IDE features like hover documentation and quick help
- Create a consistent documentation style across the project
- Help new team members get up to speed quickly
