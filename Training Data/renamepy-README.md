# rename.py
Mayank Mali

the rename.py tool is a command line program. 
It takes a directory of only files and turns them into an set of zero-padded, numbered files.

Run with python:

```$ python rename.py -h```

For example,

```$ python rename.py ./folder1/ ./folder2/img_%04d.jpg```

takes all the files from `./folder1/` and renames each to `./folder2/` as `img_0001`, `img_0002`, ... etc.

To start numbering files with a non-zero number:

```$ python rename.py ./folder1/ ./folder2/img_%04d.jpg --start 512```

This does the same as before, but now the files are  `img_0512`, `img_0513`, ... etc.