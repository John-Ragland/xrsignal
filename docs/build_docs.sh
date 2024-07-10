sphinx-apidoc -o "api/" --module-first --no-toc --force "../src/xrsignal"
sphinx-build -b html . "_build/"