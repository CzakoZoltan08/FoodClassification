# -*- coding: utf-8 -*-
from werkzeug.utils import secure_filename

from os import path


def save_file(file, folder_path):
    filename = secure_filename(file.filename)
    filepath = path.join(
            folder_path,
            'static',
            'images',
            filename)
    file.save(filepath)

    return filepath
