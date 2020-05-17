# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import inspect
import os
import shutil
import sys
import pypandoc

import rising_sphinx_theme

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.dirname(os.path.dirname(PATH_HERE))
sys.path.insert(0, os.path.abspath(PATH_ROOT))

import rising  # noqa: E402

for md in ['CONTRIBUTING.md', 'README.md']:
    shutil.copy(os.path.join(PATH_ROOT, md), os.path.join(PATH_HERE, md.lower()))

converted_readme = pypandoc.convert_file('readme.md', 'rst').split('\n')
os.remove('readme.md')

rst_file = []
skip = False

# skip problematic parts 
for line in converted_readme:
    if any([line.startswith(x) for x in ['.. container::' ,'   |PyPI|', 'Why another framework?', '.. |PyPI|', '|PyPi|']]):
        skip = True
    elif any([line.startswith(x) for x in ['What is ``rising``?', 'Installation', '.. |DefaultAugmentation|']]):
        skip = False

    if not skip:
        rst_file.append(line.replace('docs/source/images', 'images').replace('.svg', '.png'))
    
with open('getting_started.rst', 'w') as f:
    f.write('\n'.join(rst_file))

# -- Project information -----------------------------------------------------

project = 'rising'
copyright = rising.__copyright__
author = rising.__author__

# The short X.Y version
version = rising.__version__
# The full version, including alpha/beta/rc tags
release = rising.__version__


IS_REALESE = not ('+' in version or 'dirty' in version or len(version.split('.')) > 3)

# Options for the linkcode extension
# ----------------------------------
github_user = 'PhoenixDL'
github_repo = project

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

needs_sphinx = '2.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.linkcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax'
]

# napoleon_use_ivar = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

if IS_REALESE:
    templates_path = ['_templates_stable']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = ['.rst', '.md', '.ipynb']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
#    '.ipynb': 'nbsphinx',
}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# http://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# html_theme = 'bizstyle'
# https://sphinx-themes.org
# html_theme = 'pytorch_sphinx_theme'
# html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme = 'rising_sphinx_theme'
html_theme_path = [rising_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'pytorch_project': 'docs',
    'canonical_url': 'https://rising.rtfd.io',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

html_logo = 'images/logo/rising_logo.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + '-doc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',

    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project + '.tex', project + ' Documentation', author, 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, project, project + ' Documentation', [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, project, project + ' Documentation', author, project,
     'One line description of project.', 'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
    'dill': ('https://dill.rtfd.io/en/stable', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Disable docstring inheritance
autodoc_inherit_docstrings = True

# https://github.com/rtfd/readthedocs.org/issues/1139
# I use sphinx-apidoc to auto-generate API documentation for my project.
# Right now I have to commit these auto-generated files to my repository
# so that RTD can build them into HTML docs. It'd be cool if RTD could run
# sphinx-apidoc for me, since it's easy to forget to regen API docs
# and commit them to my repo after making changes to my code.

PACKAGES = [
    rising.__name__,
]

# Prolog and epilog to each notebook: https://nbsphinx.readthedocs.io/en/0.7.0/prolog-and-epilog.html

ENABLE_DOWNLOAD_LINK = True

nbsphinx_kernel_name = 'python3'

github_path = r'https://github.com/%s/%s/blob/master/notebooks/{{ env.doc2path(env.docname, base=None) }}' % (github_user, github_repo)
colab_path = github_path.replace('https://github.com', 'https://colab.research.google.com/github')
nbsphinx_execute = 'never'

nb_suffix = 'notebooks'
nb_doc_path = PATH_HERE # os.path.join(PATH_HERE, nb_suffix)
os.makedirs(nb_doc_path, exist_ok=True)
nb_path = os.path.join(PATH_ROOT, nb_suffix)

for item in os.listdir(nb_path):
    if os.path.isfile(os.path.join(nb_path, item)) and item.endswith('.ipynb'):
        shutil.copy2(os.path.join(nb_path, item),
                     os.path.join(nb_doc_path, item))


if ENABLE_DOWNLOAD_LINK:
    nbsphinx_prolog = r"""

    .. raw:: html

            <div class="pytorch-call-to-action-links">
                <a href="%s">
                <div id="google-colab-link">
                <img class="call-to-action-img" src="_static/images/pytorch-colab.svg"/>
                <div class="call-to-action-desktop-view">Run in Google Colab</div>
                <div class="call-to-action-mobile-view">Colab</div>
                </div>
                </a>
                <a href="%s" download>
                <div id="download-notebook-link">
                <img class="call-to-action-notebook-img" src="_static/images/pytorch-download.svg"/>
                <div class="call-to-action-desktop-view">Download Notebook</div>
                <div class="call-to-action-mobile-view">Notebook</div>
                </div>
                </a>
                <a href="%s">
                <div id="github-view-link">
                <img class="call-to-action-img" src="_static/images/pytorch-github.svg"/>
                <div class="call-to-action-desktop-view">View on GitHub</div>
                <div class="call-to-action-mobile-view">GitHub</div>
                </div>
                </a>
            </div>

    """ % (colab_path, r"{{ env.doc2path(env.docname, base=None) }}", github_path)

else:
    nbsphinx_prolog = r"""

    .. raw:: html

            <div class="pytorch-call-to-action-links">
                <a href="%s">
                <div id="google-colab-link">
                <img class="call-to-action-img" src="_static/images/pytorch-colab.svg"/>
                <div class="call-to-action-desktop-view">Run in Google Colab</div>
                <div class="call-to-action-mobile-view">Colab</div>
                </div>
                </a>
                <a href="%s">
                <div id="github-view-link">
                <img class="call-to-action-img" src="_static/images/pytorch-github.svg"/>
                <div class="call-to-action-desktop-view">View on GitHub</div>
                <div class="call-to-action-mobile-view">GitHub</div>
                </div>
                </a>
            </div>
    """ % (colab_path, github_path)

from docutils import nodes
from sphinx.util.docfields import TypedField
from sphinx import addnodes
import sphinx.ext.doctest

# Without this, doctest adds any example with a `>>>` as a test
doctest_test_doctest_blocks = ''
doctest_default_flags = sphinx.ext.doctest.doctest.ELLIPSIS

# Ignoring Third-party packages
# https://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule

MOCK_REQUIRE_PACKAGES = []
with open(os.path.join(PATH_ROOT, 'requirements', 'install.txt'), 'r') as fp:
    for ln in fp.readlines():
        found = [ln.index(ch) for ch in list(',=<>#') if ch in ln]
        pkg = ln[:min(found)] if found else ln
        if pkg.rstrip():
            MOCK_REQUIRE_PACKAGES.append(pkg.rstrip())

with open(os.path.join(PATH_ROOT, 'requirements', 'install_async.txt'), 'r') as fp:
    for ln in fp.readlines():
        found = [ln.index(ch) for ch in list(',=<>#') if ch in ln]
        pkg = ln[:min(found)] if found else ln
        if pkg.rstrip():
            MOCK_REQUIRE_PACKAGES.append(pkg.rstrip())

# TODO: better parse from package since the import name and package name may differ
MOCK_MANUAL_PACKAGES = [
    'torch',
    'torchvision',
    'numpy',
    'dill'
]
autodoc_mock_imports = MOCK_REQUIRE_PACKAGES + MOCK_MANUAL_PACKAGES


# Resolve function
# This function is used to populate the (source) links in the API
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        fname = inspect.getsourcefile(obj)
        # https://github.com/rtfd/readthedocs.org/issues/5735
        if any([s in fname for s in ('readthedocs', 'rtfd', 'checkouts')]):
            # /home/docs/checkouts/readthedocs.org/user_builds/pytorch_lightning/checkouts/
            #  devel/pytorch_lightning/utilities/cls_experiment.py#L26-L176
            path_top = os.path.abspath(os.path.join('..', '..', '..'))
            fname = os.path.relpath(fname, start=path_top)
        else:
            # Local build, imitate master
            fname = 'master/' + os.path.relpath(fname, start=os.path.abspath('..'))
        source, lineno = inspect.getsourcelines(obj)
        return fname, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = '%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    # import subprocess
    # tag = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
    #                        universal_newlines=True).communicate()[0][:-1]
    branch = filename.split('/')[0]
    # do mapping from latest tags to master
    branch = {'latest': 'master', 'stable': 'master'}.get(branch, branch)
    filename = '/'.join([branch] + filename.split('/')[1:])
    return "https://github.com/%s/%s/blob/%s" \
           % (github_user, github_repo, filename)


autodoc_member_order = 'groupwise'
autoclass_content = 'both'
# the options are fixed and will be soon in release,
#  see https://github.com/sphinx-doc/sphinx/issues/5459
autodoc_default_options = {
    'members': None,
    'methods': None,
    # 'attributes': None,
    'special-members': '__call__',
    'exclude-members': '_abc_impl',
    'show-inheritance': True,
    'private-members': True,
    'noindex': True,
}

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_add_permalinks = "¶"

# True to prefix each section label with the name of the document it is in, followed by a colon.
#  For example, index:Introduction for a section called Introduction that appears in document index.rst.
#  Useful for avoiding ambiguity when the same section heading appears in different documents.
# http://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True

html_show_sphinx = False
