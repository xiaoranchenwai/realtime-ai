# Contributing / 贡献指南

Thank you for your interest in contributing to this project!

感谢你对本项目的关注！

## Getting Started / 开始

### Prerequisites / 前置要求

- Python 3.8+
- pip

### Development Setup / 开发环境设置

1. Clone the repository / 克隆仓库

```bash
git clone https://github.com/chicogong/realtime-ai.git
cd realtime-ai
```

2. Create virtual environment / 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
pip install -e ".[dev]"  # Install dev dependencies
```

4. Set up pre-commit hooks / 设置预提交钩子

```bash
pre-commit install
```

5. Copy environment file / 复制环境变量文件

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Code Style / 代码风格

This project uses:

- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

Run formatting:

```bash
black .
isort .
```

Run linting:

```bash
ruff check .
mypy .
```

## Making Changes / 提交修改

1. Create a new branch / 创建新分支

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes / 进行修改

3. Run tests and checks / 运行测试和检查

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type check
mypy .
```

4. Commit your changes / 提交修改

```bash
git add .
git commit -m "feat: add your feature description"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance

5. Push and create PR / 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Questions / 问题

If you have any questions, feel free to open an issue.

如有任何问题，欢迎提 Issue。
