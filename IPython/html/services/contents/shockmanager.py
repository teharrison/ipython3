"""A contents manager that uses: 
1. local file system for file and directory storage
2. Shock for notebook storage

https://github.com/MG-RAST/Shock"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import dateutil.parser
import requests
import json
import uuid
from collections import defaultdict

import base64
import io
import os
import glob
import shutil

from tornado import web

from .manager import ContentsManager
from IPython import nbformat
from IPython.utils.io import atomic_writing
from IPython.utils.path import ensure_dir_exists
from IPython.utils.traitlets import Unicode, Bool, TraitError
from IPython.utils.py3compat import getcwd
from IPython.utils import tz
from IPython.html.utils import is_hidden, to_os_path, url_path_join


class ShockContentsManager(ContentsManager):
    shock_url = Unicode('', config=True, help='Shock server url')
    token     = Unicode('', config=True, help='OAuth bearer token (OAuth v2.0)')
    node_type = 'ipynb'
    nb_list   = {}
    headers   = {}
    root_dir  = Unicode(getcwd(), config=True)
    
    def __init__(self, **kwargs):
        """verify Shock Authentication credintals, set auth header"""
        super(ShockContentsManager, self).__init__(**kwargs)
        if not self.shock_url:
            raise web.HTTPError(412, u"Missing Shock server URI.")
        if not self.token:
            if 'KB_AUTH_TOKEN' in os.environ:
                self.token = os.environ['KB_AUTH_TOKEN']
            else:
                raise web.HTTPError(412, u"Missing credintals for Shock Authentication.")
        self.headers = {'headers': {'Authorization': 'OAuth %s'%self.token}}
        self._get_notebook_list()
        self._set_notebook_list()
    
    save_script = Bool(False, config=True, help='DEPRECATED, IGNORED')
    def _save_script_changed(self):
        self.log.warn("""
        Automatically saving notebooks as scripts has been removed.
        Use `ipython nbconvert --to python [notebook]` instead.
        """)

    def _root_dir_changed(self, name, old, new):
        """Do a bit of validation of the root_dir."""
        if not os.path.isabs(new):
            # If we receive a non-absolute path, make it absolute.
            self.root_dir = os.path.abspath(new)
            return
        if not os.path.isdir(new):
            raise TraitError("%r is not a directory" % new)

    checkpoint_dir = Unicode('.ipynb_checkpoints', config=True,
        help="""The directory name in which to keep file checkpoints

        This is a path relative to the file's own directory.

        By default, it is .ipynb_checkpoints
        """
    )

    # name in shock w/o extension
    def _strip_ext(self, name):
        if name.endswith('.ipynb'):
            return name[:-6]
        else:
            return name

    # notebooks need .ipynb extension
    def _add_ext(self, name):
        if not name.endswith('.ipynb'):
            return name+'.ipynb'
        else:
            return name

    def _get_notebook_list(self):
        """load the notebook names from Shock.
        The name is stored as Shock metadata.
            1. Skip nb nodes with no files
            2. Skip nb nodes tagged as 'deleted'
            3. If multiple nb with same name, get latest timestamp
        """
        self.nb_list = {}
        nb_vers = defaultdict(list)
        
        query_path = '?query&type='+self.node_type+'&format=json&limit=0'
        query_result = self._get_shock_node(query_path, 'json')
        
        if query_result is not None:
            for node in query_result:
                # node need file and name and original id
                if node['file']['size'] and ('original' in node['attributes']) and node['attributes']['original'] and ('name' in node['attributes']) and node['attributes']['name']:
                    # group by original to include renaming
                    nb_vers[ node['attributes']['original'] ].append(node)

        # only get listing of latest for each notebook name set
        for orig in nb_vers.keys():
            nodes = sorted(nb_vers[orig], key=lambda x: x['attributes']['last_modified'], reverse=True)
            # if latest is flaged deleted - don't show
            if ('deleted' in nodes[0]['attributes']) and nodes[0]['attributes']['deleted']:
                continue
            # notebooks need .ipynb extension
            name = self._add_ext(nodes[0]['attributes']['name'])
            self.nb_list[name] = nodes[0]

    def _set_notebook_list(self):
        # create notebook stubs in root dir
        self.log.info("Retrieving notebooks from Shock:")
        for name in self.nb_list.keys():
            path = os.path.join(self.root_dir, name)
            self.log.info("%s -> %s", self._strip_ext(name), path)
            open(path, 'w').close()
    
    def _copy(self, src, dest):
        """copy src to dest

        like shutil.copy2, but log errors in copystat
        """
        shutil.copyfile(src, dest)
        try:
            shutil.copystat(src, dest)
        except OSError as e:
            self.log.debug("copystat on %s failed", dest, exc_info=True)

    def _get_os_path(self, path):
        """Given an API path, return its file system path.

        Parameters
        ----------
        path : string
            The relative API path to the named file.

        Returns
        -------
        path : string
            Native, absolute OS path to for a file.
        """
        return to_os_path(path, self.root_dir)

    def dir_exists(self, path):
        """Does the API-style path refer to an extant directory?

        API-style wrapper for os.path.isdir

        Parameters
        ----------
        path : string
            The path to check. This is an API path (`/` separated,
            relative to root_dir).

        Returns
        -------
        exists : bool
            Whether the path is indeed a directory.
        """
        path = path.strip('/')
        os_path = self._get_os_path(path=path)
        return os.path.isdir(os_path)

    def is_hidden(self, path):
        """Does the API style path correspond to a hidden directory or file?

        Parameters
        ----------
        path : string
            The path to check. This is an API path (`/` separated,
            relative to root_dir).

        Returns
        -------
        hidden : bool
            Whether the path exists and is hidden.
        """
        path = path.strip('/')
        os_path = self._get_os_path(path=path)
        return is_hidden(os_path, self.root_dir)

    def file_exists(self, path):
        """Returns True if the file exists, else returns False.

        API-style wrapper for os.path.isfile

        Parameters
        ----------
        path : string
            The relative path to the file (with '/' as separator)

        Returns
        -------
        exists : bool
            Whether the file exists.
        """
        path = path.strip('/')
        os_path = self._get_os_path(path)
        return os.path.isfile(os_path)

    def exists(self, path):
        """Returns True if the path exists, else returns False.

        API-style wrapper for os.path.exists

        Parameters
        ----------
        path : string
            The API path to the file (with '/' as separator)

        Returns
        -------
        exists : bool
            Whether the target exists.
        """
        path = path.strip('/')
        os_path = self._get_os_path(path=path)
        return os.path.exists(os_path)

    def _base_model(self, path):
        """Build the common base of a contents model"""
        os_path = self._get_os_path(path)
        info = os.stat(os_path)
        last_modified = tz.utcfromtimestamp(info.st_mtime)
        created = tz.utcfromtimestamp(info.st_ctime)
        # Create the base model.
        model = {}
        model['name'] = path.rsplit('/', 1)[-1]
        model['path'] = path
        model['last_modified'] = last_modified
        model['created'] = created
        model['content'] = None
        model['format'] = None
        return model

    def _dir_model(self, path, content=True):
        """Build a model for a directory

        if content is requested, will include a listing of the directory
        if is root_dir will refresh notebook listing from shock
        """
        os_path = self._get_os_path(path)

        four_o_four = u'directory does not exist: %r' % os_path

        if not os.path.isdir(os_path):
            raise web.HTTPError(404, four_o_four)
        elif is_hidden(os_path, self.root_dir):
            self.log.info("Refusing to serve hidden directory %r, via 404 Error",
                os_path
            )
            raise web.HTTPError(404, four_o_four)

        model = self._base_model(path)
        model['type'] = 'directory'
        if content:
            model['content'] = contents = []
            os_dir = self._get_os_path(path)
            # refresh notebooks
            if os_dir == self.root_dir:
                self._get_notebook_list()
                self._set_notebook_list()
            # get listing
            for name in os.listdir(os_dir):
                os_path = os.path.join(os_dir, name)
                # skip over broken symlinks in listing
                if not os.path.exists(os_path):
                    self.log.warn("%s doesn't exist", os_path)
                    continue
                elif not os.path.isfile(os_path) and not os.path.isdir(os_path):
                    self.log.debug("%s not a regular file", os_path)
                    continue
                if self.should_list(name) and not is_hidden(os_path, self.root_dir):
                    contents.append(self.get(
                        path='%s/%s' % (path, name),
                        content=False)
                    )

            model['format'] = 'json'
        return model

    def _file_model(self, path, content=True, format=None):
        """Build a model for a file

        if content is requested, include the file contents.

        format:
          If 'text', the contents will be decoded as UTF-8.
          If 'base64', the raw bytes contents will be encoded as base64.
          If not specified, try to decode as UTF-8, and fall back to base64
        """
        model = self._base_model(path)
        model['type'] = 'file'
        if content:
            os_path = self._get_os_path(path)
            if not os.path.isfile(os_path):
                # could be FIFO
                raise web.HTTPError(400, "Cannot get content of non-file %s" % os_path)
            with io.open(os_path, 'rb') as f:
                bcontent = f.read()

            if format != 'base64':
                try:
                    model['content'] = bcontent.decode('utf8')
                except UnicodeError as e:
                    if format == 'text':
                        raise web.HTTPError(400, "%s is not UTF-8 encoded" % path)
                else:
                    model['format'] = 'text'

            if model['content'] is None:
                model['content'] = base64.encodestring(bcontent).decode('ascii')
                model['format'] = 'base64'
        return model

    def _notebook_model(self, path, content=True):
        """Build a notebook model from shock node

        if content is requested, the notebook content will be populated
        as a JSON structure (not double-serialized)
        """
        self._get_notebook_list()
        model = self._base_model(path)
        if model['name'] not in self.nb_list:
            raise web.HTTPError(400, u"Unavailable Notebook: %s" % (model['name']))
        
        model['type'] = 'notebook'
        model['last_modified'] = dateutil.parser.parse(self.nb_list[model['name']]['attributes']['last_modified'])
        model['last_modified'].replace(tzinfo=tz.UTC)
        model['created'] = dateutil.parser.parse(self.nb_list[model['name']]['attributes']['created'])
        model['created'].replace(tzinfo=tz.UTC)
        
        if content:
            try:
                node_path = "/%s?download" %self.nb_list[model['name']]['id']
                node_data = self._get_shock_node(node_path, 'data')
            except:
                raise web.HTTPError(500, u'Notebook cannot be read')
            try:
                nb = nbformat.reads(node_data, as_version=4)
            except Exception as e:
                raise web.HTTPError(400, u"Unreadable Notebook: %s %r" % (model['name'], e))
            self.mark_trusted_cells(nb, model['name'])
            model['content'] = nb
            model['format'] = 'json'
            self.validate_notebook_model(model)
        return model

    def get(self, path, content=True, type_=None, format=None):
        """ Takes a path for an entity and returns its model

        Parameters
        ----------
        path : str
            the API path that describes the relative path for the target
        content : bool
            Whether to include the contents in the reply
        type_ : str, optional
            The requested type - 'file', 'notebook', or 'directory'.
            Will raise HTTPError 400 if the content doesn't match.
        format : str, optional
            The requested format for file contents. 'text' or 'base64'.
            Ignored if this returns a notebook or directory model.

        Returns
        -------
        model : dict
            the contents model. If content=True, returns the contents
            of the file or directory as well.
        """
        path = path.strip('/')

        if not self.exists(path):
            raise web.HTTPError(404, u'No such file or directory: %s' % path)

        os_path = self._get_os_path(path)
        if os.path.isdir(os_path):
            if type_ not in (None, 'directory'):
                raise web.HTTPError(400,
                                u'%s is a directory, not a %s' % (path, type_))
            model = self._dir_model(path, content=content)
        elif type_ == 'notebook' or (type_ is None and path.endswith('.ipynb')):
            model = self._notebook_model(path, content=content)
        else:
            if type_ == 'directory':
                raise web.HTTPError(400,
                                u'%s is not a directory')
            model = self._file_model(path, content=content, format=format)
        return model

    def _save_notebook(self, os_path, model, path):
        """save a notebook to shock"""
        # Get name
        name = path.rsplit('/', 1)[-1]
        # Get attributes
        attr = {}
        attr['name'] = self._strip_ext(name)
        attr['type'] = self.node_type
        attr['format'] = 'json'
        attr['last_modified'] = tz.utcnow().isoformat()
        # creation timestamp
        if 'created' in model:
            attr['created'] = model['created'].isoformat()
        elif name in self.nb_list:
            attr['created'] = self.nb_list[name]['attributes']['created']
        else:
            attr['created'] = attr['last_modified']
        # original id
        if name in self.nb_list:
            attr['original'] = self.nb_list[name]['attributes']['original']
        else:
            attr['original'] = str(uuid.uuid4())
        attr_str = json.dumps(attr)
        # Get the notebook content
        nb = nbformat.from_dict(model['content'])
        self.check_and_sign(nb, name)
        nb_str = nbformat.writes(nb, version=nbformat.NO_CONVERT)
        # Save to shock
        try:
            self.log.debug("Saving %s to Shock", name)
            node = self._post_shock_node(name, nb_str, attr_str)
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while saving notebook: %s' %e)
        # update lists
        self.nb_list[name] = node
        open(os_path, 'w').close()

    def _save_file(self, os_path, model, path=''):
        """save a non-notebook file"""
        fmt = model.get('format', None)
        if fmt not in {'text', 'base64'}:
            raise web.HTTPError(400, "Must specify format of file contents as 'text' or 'base64'")
        try:
            content = model['content']
            if fmt == 'text':
                bcontent = content.encode('utf8')
            else:
                b64_bytes = content.encode('ascii')
                bcontent = base64.decodestring(b64_bytes)
        except Exception as e:
            raise web.HTTPError(400, u'Encoding error saving %s: %s' % (os_path, e))
        with atomic_writing(os_path, text=False) as f:
            f.write(bcontent)

    def _save_directory(self, os_path, model, path=''):
        """create a directory"""
        if is_hidden(os_path, self.root_dir):
            raise web.HTTPError(400, u'Cannot create hidden directory %r' % os_path)
        if not os.path.exists(os_path):
            os.mkdir(os_path)
        elif not os.path.isdir(os_path):
            raise web.HTTPError(400, u'Not a directory: %s' % (os_path))
        else:
            self.log.debug("Directory %r already exists", os_path)

    def save(self, model, path=''):
        """Save the file model and return the model with no content."""
        path = path.strip('/')

        if 'type' not in model:
            raise web.HTTPError(400, u'No file type provided')
        if 'content' not in model and model['type'] != 'directory':
            raise web.HTTPError(400, u'No file content provided')

        # One checkpoint should always exist
        if self.file_exists(path) and not self.list_checkpoints(path):
            self.create_checkpoint(path)

        os_path = self._get_os_path(path)
        self.log.debug("Saving %s", os_path)
        try:
            if model['type'] == 'notebook':
                self._save_notebook(os_path, model, path)
            elif model['type'] == 'file':
                self._save_file(os_path, model, path)
            elif model['type'] == 'directory':
                self._save_directory(os_path, model, path)
            else:
                raise web.HTTPError(400, "Unhandled contents type: %s" % model['type'])
        except web.HTTPError:
            raise
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while saving file: %s %s' % (os_path, e))

        validation_message = None
        if model['type'] == 'notebook':
            self.validate_notebook_model(model)
            validation_message = model.get('message', None)

        model = self.get(path, content=False)
        if validation_message:
            model['message'] = validation_message
        return model

    def update(self, model, path):
        """Update the file's path

        For use in PATCH requests, to enable renaming a file without
        re-uploading its contents. Only used for renaming at the moment.
        """
        path = path.strip('/')
        new_path = model.get('path', path).strip('/')
        if path != new_path:
            self.rename(path, new_path)
        model = self.get(new_path, content=False)
        return model

    def _delete_notebook(self, os_path, name):
        if name not in self.nb_list:
            raise web.HTTPError(400, u"Unavailable Notebook: %s" % (name))
        # set as deleted in shock
        attr = self.nb_list[name]['attributes']
        attr['deleted'] = 1
        attr['last_modified'] = tz.utcnow().isoformat()
        attr_str = json.dumps(attr)
        try:
            self.log.debug("Deleting %s from Shock", name)
            node = self._update_shock_node(self.nb_list[name]['id'], attr_str)
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while deleting notebook: %s' %e)
        # delete from listing
        del self.nb_list[name]
        os.unlink(os_path)

    def delete(self, path):
        """Delete file at path."""
        path = path.strip('/')
        os_path = self._get_os_path(path)
        rm = os.unlink
        if os.path.isdir(os_path):
            listing = os.listdir(os_path)
            # don't delete non-empty directories (checkpoints dir doesn't count)
            if listing and listing != [self.checkpoint_dir]:
                raise web.HTTPError(400, u'Directory %s not empty' % os_path)
        elif not os.path.isfile(os_path):
            raise web.HTTPError(404, u'File does not exist: %s' % os_path)

        # clear checkpoints
        for checkpoint in self.list_checkpoints(path):
            checkpoint_id = checkpoint['id']
            cp_path = self.get_checkpoint_path(checkpoint_id, path)
            if os.path.isfile(cp_path):
                self.log.debug("Unlinking checkpoint %s", cp_path)
                os.unlink(cp_path)

        if os.path.isdir(os_path):
            self.log.debug("Removing directory %s", os_path)
            shutil.rmtree(os_path)
        elif path.endswith('.ipynb'):
            name = path.rsplit('/', 1)[-1]
            self.log.debug("Deleting Notebook %s", name)
            self._delete_notebook(os_path, name)
        else:
            self.log.debug("Unlinking file %s", os_path)
            rm(os_path)

    def _rename_notebook(self, old_name, new_name):
        if old_name not in self.nb_list:
            raise web.HTTPError(400, u"Unavailable Notebook: %s" % (old_name))
        # update attributes
        attr = self.nb_list[old_name]['attributes']
        attr['name'] = self._strip_ext(new_name)
        attr['last_modified'] = tz.utcnow().isoformat()
        attr_str = json.dumps(attr)
        # update in shock
        try:
            node = self._update_shock_node(self.nb_list[old_name]['id'], attr_str)
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while renaming notebook: %s' %e)
        # update listing
        del self.nb_list[old_name]
        self.nb_list[new_name] = node
    
    def rename(self, old_path, new_path):
        """Rename a file."""
        old_path = old_path.strip('/')
        new_path = new_path.strip('/')
        if new_path == old_path:
            return

        new_os_path = self._get_os_path(new_path)
        old_os_path = self._get_os_path(old_path)

        # Should we proceed with the move?
        if os.path.exists(new_os_path):
            raise web.HTTPError(409, u'File already exists: %s' % new_path)

        # Rename notebook
        if old_path.endswith('.ipynb'):
            self._rename_notebook(old_path, new_path)
        # Move the file
        try:
            shutil.move(old_os_path, new_os_path)
        except Exception as e:
            raise web.HTTPError(500, u'Unknown error renaming file: %s %s' % (old_path, e))

        # Move the checkpoints
        old_checkpoints = self.list_checkpoints(old_path)
        for cp in old_checkpoints:
            checkpoint_id = cp['id']
            old_cp_path = self.get_checkpoint_path(checkpoint_id, old_path)
            new_cp_path = self.get_checkpoint_path(checkpoint_id, new_path)
            if os.path.isfile(old_cp_path):
                self.log.debug("Renaming checkpoint %s -> %s", old_cp_path, new_cp_path)
                shutil.move(old_cp_path, new_cp_path)

    # Checkpoint-related utilities

    def get_checkpoint_path(self, checkpoint_id, path):
        """find the path to a checkpoint"""
        path = path.strip('/')
        parent, name = ('/' + path).rsplit('/', 1)
        parent = parent.strip('/')
        basename, ext = os.path.splitext(name)
        filename = u"{name}-{checkpoint_id}{ext}".format(
            name=basename,
            checkpoint_id=checkpoint_id,
            ext=ext,
        )
        os_path = self._get_os_path(path=parent)
        cp_dir = os.path.join(os_path, self.checkpoint_dir)
        ensure_dir_exists(cp_dir)
        cp_path = os.path.join(cp_dir, filename)
        return cp_path

    def get_checkpoint_model(self, checkpoint_id, path):
        """construct the info dict for a given checkpoint"""
        path = path.strip('/')
        cp_path = self.get_checkpoint_path(checkpoint_id, path)
        stats = os.stat(cp_path)
        last_modified = tz.utcfromtimestamp(stats.st_mtime)
        info = dict(
            id = checkpoint_id,
            last_modified = last_modified,
        )
        return info

    # public checkpoint API

    def create_checkpoint(self, path):
        """Create a checkpoint from the current state of a file"""
        path = path.strip('/')
        if not self.file_exists(path):
            raise web.HTTPError(404)
        src_path = self._get_os_path(path)
        # only the one checkpoint ID:
        checkpoint_id = u"checkpoint"
        cp_path = self.get_checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        self._copy(src_path, cp_path)

        # return the checkpoint info
        return self.get_checkpoint_model(checkpoint_id, path)

    def list_checkpoints(self, path):
        """list the checkpoints for a given file

        This contents manager currently only supports one checkpoint per file.
        """
        path = path.strip('/')
        checkpoint_id = "checkpoint"
        os_path = self.get_checkpoint_path(checkpoint_id, path)
        if not os.path.exists(os_path):
            return []
        else:
            return [self.get_checkpoint_model(checkpoint_id, path)]


    def restore_checkpoint(self, checkpoint_id, path):
        """restore a file to a checkpointed state"""
        path = path.strip('/')
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        nb_path = self._get_os_path(path)
        cp_path = self.get_checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(cp_path):
            self.log.debug("checkpoint file does not exist: %s", cp_path)
            raise web.HTTPError(404,
                u'checkpoint does not exist: %s@%s' % (path, checkpoint_id)
            )
        # ensure notebook is readable (never restore from an unreadable notebook)
        if cp_path.endswith('.ipynb'):
            with io.open(cp_path, 'r', encoding='utf-8') as f:
                nbformat.read(f, as_version=4)
        self._copy(cp_path, nb_path)
        self.log.debug("copying %s -> %s", cp_path, nb_path)

    def delete_checkpoint(self, checkpoint_id, path):
        """delete a file's checkpoint"""
        path = path.strip('/')
        cp_path = self.get_checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(cp_path):
            raise web.HTTPError(404,
                u'Checkpoint does not exist: %s@%s' % (path, checkpoint_id)
            )
        self.log.debug("unlinking %s", cp_path)
        os.unlink(cp_path)

    def info_string(self):
        return "Serving notebooks from Shock storage: %s" % self.shock_url

    def get_kernel_path(self, path, model=None):
        """Return the initial working dir a kernel associated with a given notebook"""
        if '/' in path:
            parent_dir = path.rsplit('/', 1)[0]
        else:
            parent_dir = ''
        return self._get_os_path(parent_dir)
    
    # shock related utilities
    
    def _get_shock_node(self, path, format):
        url = self.shock_url+'/node'+path
        try:
            rget = requests.get(url, **self.headers)
        except Exception as e:
            raise web.HTTPError(504, u'Unable to connect to Shock server %s: %s' %(url, e))
        if not (rget.ok and rget.text):
            raise web.HTTPError(504, u'Unable to connect to Shock server %s: %s' %(url, rget.raise_for_status()))
        if format == 'json':
            rj = rget.json()
            if not (rj and isinstance(rj, dict) and all([key in rj for key in ['status','data','error']])):
                raise web.HTTPError(415, u'Return data not valid Shock format')
            if rj['error']:
                raise web.HTTPError(rj['status'], 'Shock error: '+rj['error'])
            return rj['data']
        else:
            return rget.text
    
    def _post_shock_node(self, name, data, attr):
        url = self.shock_url+'/node'
        data_hdl = io.StringIO(data)
        attr_hdl = io.StringIO(attr)
        files = {
            "upload": (name, data_hdl),
            "attributes": ('%s.json'%name, attr_hdl)
        }
        try:
            kwargs = {'files': files}
            kwargs.update(self.headers)
            rpost = requests.post(url, **kwargs)
            rj = rpost.json()
        except Exception as e:
            raise web.HTTPError(504, u'Unable to connect to Shock server %s: %s' %(url, e))
        if not (rpost.ok and rj and isinstance(rj, dict) and all([key in rj for key in ['status','data','error']])):
            raise web.HTTPError(500, u'Unable to POST to Shock server %s: %s' %(url, rpost.raise_for_status()))
        if rj['error']:
            raise web.HTTPError(rj['status'], 'Shock error: '+rj['error'])
        return rj['data']

    def _update_shock_node(self, node, attr):
        url = self.shock_url+'/node/'+node
        attr_hdl = io.StringIO(attr)
        files = { "attributes": ('%s.json'%node, attr_hdl) }
        try:
            kwargs = {'files': files}
            kwargs.update(self.headers)
            rpost = requests.put(url, **kwargs)
            rj = rpost.json()
        except Exception as e:
            raise web.HTTPError(504, u'Unable to connect to Shock server %s: %s' %(url, e))
        if not (rpost.ok and rj and isinstance(rj, dict) and all([key in rj for key in ['status','data','error']])):
            raise web.HTTPError(500, u'Unable to POST to Shock server %s: %s' %(url, rpost.raise_for_status()))
        if rj['error']:
            raise web.HTTPError(rj['status'], 'Shock error: '+rj['error'])
        return rj['data']

