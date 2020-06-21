# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'PySoundTool'
copyright = '2020, Aislyn Rose'
author = 'Aislyn Rose'

# The full version, including alpha/beta/rc tags
release = '0.1.0b'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    'sphinx.ext.autosummary']

sphinx_gallery_conf = {
    'examples_dirs': 'examples/',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': '/plot_',
    'ignore_pattern': r'__init__\.py',
        'reference_url': {
            'sphinx_gallery': None,
            'numpy': 'http://docs.scipy.org/doc/numpy/',
            'np': 'http://docs.scipy.org/doc/numpy/',
            'scipy': 'http://docs.scipy.org/doc/scipy/reference',
            'matplotlib': 'https://matplotlib.org/',
            'sklearn': 'https://scikit-learn.org/stable',
            'soundfile': 'https://pysoundfile.readthedocs.io/en/latest/',
            'sf': 'https://pysoundfile.readthedocs.io/en/latest/',
            'librosa' : 'https://librosa.org/librosa/'
        }
    }
        
autosummary_generate = True 

# Generate plots for example sections
numpydoc_use_plots = True
        
# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'autolink'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

#Napoleon settings
napoleon_numpy_docstring = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------


# -- Options for HTML output -------------------------------------------------
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

html_use_modindex = True




# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pysoundtooldoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    #The paper size ('letterpaper' or 'a4paper').
    
    'papersize': 'letterpaper',

    #The font size ('10pt', '11pt' or '12pt').
    
    'pointsize': '10pt',

    #Additional stuff for the LaTeX preamble.
    
    'preamble': '',

    #Latex figure (float) alignment
    
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
master_doc = 'index'
latex_documents = [
    (master_doc, 'pysoundtool.tex', u'PySoundTool Docs',
     u'Aislyn Rose', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'PySoundTool', u'Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'PySoundTool', u'Documentation',
     author, 'PySoundTool', 
     'A framework for exploring and experimenting with acoustics and deep learning.',
     'Miscellaneous'),
]



# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']



# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# Establishes order that modules are listed; default alphabetical.
autodoc_member_order = 'bysource'
