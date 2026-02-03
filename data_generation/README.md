Create a DIFFERENT venv for data_generation. Since some packages aren't compatible for both moshi and dia.

Note: Install DIA2 with the following command. It seems that some directories don't exist in the pip wheel.

```bash
pip uninstall dia2 -y
git clone https://github.com/nari-labs/dia2.git /tmp/dia2
pip install -e /tmp/dia2
```
In this way, you have a complete repository of dia2. 