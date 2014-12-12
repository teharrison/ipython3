"""A contents manager that uses: 
1. local file system for file and directory storage
2. Workspace for notebook storage

https://github.com/kbase/workspace_deluxe"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import dateutil.parser
import urllib.request, urllib.error, urllib.parse
import re
import json
import random
from urllib.error import URLError, HTTPError
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

class JSONObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

class WorkspaceContentsManager(ContentsManager):
    ws_url   = Unicode('', config=True, help='Workspace server url')
    token    = Unicode('', config=True, help='OAuth bearer token (OAuth v2.0)')
    root_dir = Unicode(getcwd(), config=True)
    nb_list  = {}
    ws_type  = "KBaseNarrative.Narrative"
    wsid_re  = re.compile(r"^(kb\|)?ws\.(\d+)\.obj\.(\d+)(\.ver\.(\d+))?$")
    
    def __init__(self, **kwargs):
        """verify Authentication credintals, set auth header"""
        super(ShockContentsManager, self).__init__(**kwargs)
        if not self.ws_url:
            raise web.HTTPError(412, u"Missing Workspace server URI.")
        if not self.token:
            if 'KB_AUTH_TOKEN' in os.environ:
                self.token = os.environ['KB_AUTH_TOKEN']
            else:
                raise web.HTTPError(412, u"Missing credintals for Workspace Authentication.")
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

    # name in workspace w/o extension
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
        """load the notebook names from Workspace.
        The ID is ws.<ws_id>.obj.<obj_id>
        """
        self.nb_list = {}
        nb_objs = self._get_ws_list(self.ws_type)
        for info in nb_objs:
            nbid = ".".join(['ws', info[6], 'obj', info[0]])
            self.nb_list[nbid] = info

    def _set_notebook_list(self):
        # create notebook stubs in root dir
        self.log.info("Retrieving notebooks from Workspace:")
        for nbid in self.nb_list.keys():
            name = self._nb_name_from_id(nbid)
            path = os.path.join(self.root_dir, self._add_ext(name))
            self.log.info("%s -> %s", name, path)
            open(path, 'w').close()

    def _nb_name_from_id(self, nbid):
        if nbid in self.nb_list:
            return nb_list[nbid][1]
        else:
            # missing
            raise web.HTTPError(404, u'Unavailable Notebook: %s'%nbid)
    
    def _nb_id_from_name(self, name, missing_ok=False):
        name = self._strip_ext(name)
        for nbid, info in self.nb_list.items():
            if info[1] == name:
                return nbid
        # missing
        if missing_ok:
            return None
        else:
            raise web.HTTPError(404, u'Unavailable Notebook: %s'%name)

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
        """Build a notebook model from workspace object

        if content is requested, the notebook content will be populated
        as a JSON structure (not double-serialized)
        """
        self._get_notebook_list()
        model = self._base_model(path)
        nb_id = self._nb_id_from_name(model['name'])
        model['type'] = 'notebook'
        model['last_modified'] = dateutil.parser.parse(self.nb_list[nb_id][3])
        model['last_modified'].replace(tzinfo=tz.UTC)
        
        if content:
            try:
                nb_obj = self._get_ws_object(nb_id)
                nb_data = json.dumps(nb_obj['data'])
                self.nb_list[nb_id] = nb_obj['info']
                model['created'] = dateutil.parser.parse(nb_obj['created'])
                model['created'].replace(tzinfo=tz.UTC)
            except:
                raise web.HTTPError(500, u'Notebook cannot be read: %s'%(model['name']))
            try:
                nb = nbformat.reads(nb_data, as_version=4)
            except Exception as e:
                raise web.HTTPError(400, u"Unreadable Notebook: %s %r"%(model['name'], e))
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

    def _new_notebook(self, name):
        # can only save it if have workspace name
        if not 'KB_WORKSPACE_ID' in os.environ:
            raise web.HTTPError(404, u'Unable to create new notebook, missing workspace name')
        # Get metadata
        wsname = os.environ['wsname']
        meta = {
            'ws_name': wsname,
            'description': '',
            'name': self._strip_ext(name),
            'data_dependencies': "[]",
            'format': 'ipynb',
            'type': 'Narrative',
            'creator': self._parse_token(self.token)['un']
        }
        # Get the notebook content
        nb = nbformat.from_dict(model['content'])
        self.check_and_sign(nb, name)
        # save it
        try:
            self.log.debug("Saving new notebook %s to Workspace", name)
            info = self._new_ws_obj(name, wsname, meta, nb)
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while saving notebook: %s' %e)
        nbid = ".".join(['ws', info[6], 'obj', info[0]])
        return nbid, info
    
    def _save_notebook(self, os_path, model, path):
        """save a notebook to workspace"""
        # Get name
        name = path.rsplit('/', 1)[-1]
        nbid = self._nb_id_from_name(name, missing_ok=True)
        
        # this is new
        if nbid is None:
            nbid, info = self._new_notebook(self._strip_ext(name))
        # save existing
        else:
            # Get metadata
            meta = self.nb_list[nbid][10]
            meta['name'] = self._strip_ext(name)
            meta['format'] = 'ipynb'
            meta['type'] = 'Narrative'
            # Get the notebook content
            nb = nbformat.from_dict(model['content'])
            self.check_and_sign(nb, name)
            # Save to workspace
            try:
                self.log.debug("Saving %s (%s) to Workspace", self._strip_ext(name), nbid)
                info = self._save_ws_obj(nbid, meta, nb)
            except Exception as e:
                raise web.HTTPError(400, u'Unexpected error while saving notebook: %s' %e)
        
        # update lists
        self.nb_list[nbid] = info
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
        # Get name
        name = path.rsplit('/', 1)[-1]
        nbid = self._nb_id_from_name(name)
        # delete from workspace
        try:
            self.log.debug("Deleting %s (%s) from Workspace", self._strip_ext(name), nbid)
            self._delete_ws_obj(nbid)
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while deleting notebook: %s' %e)
        # delete from listing
        del self.nb_list[nbid]
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
        old_name = old_name.rsplit('/', 1)[-1]
        new_name = new_name.rsplit('/', 1)[-1]
        nbid = self._nb_id_from_name(old_name)
        # rename in workspace
        try:
            info = self._rename_ws_obj(nbid, self._strip_ext(new_name))
        except Exception as e:
            raise web.HTTPError(400, u'Unexpected error while renaming notebook: %s' %e)
        # update info
        self.nb_list[nbid] = info
    
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
    
    # workspace related utilities
    
    def _parse_token(self, token):
        token_dict = {}
        for part in token.strip().split('|'):
            k, v = part.split('=')
            if k and v:
                token_dict[k] = v
        for item in ['un', 'SigningSubject']:
            if item not in token_dict:
                raise web.HTTPError(403, u'Invalid OAuth token structure: %s' %token)
        return token_dict
    
    def _parse_ws_id(self, id_str):
        # check for refs of the form kb|ws.1.obj.2.ver.4
        id_str = self._strip_ext(id_str)
        m = re.match(self.wsid_re, id_str)
        if m:
            x = m.groups()
            # no ver
            if x[4] is None:
                return {'wsid': x[1], 'objid': x[2]}
            else:
                return {'wsid': x[1], 'objid': x[2], 'ver': x[4]}
        else:
            raise web.HTTPError(404, u'Invalid workspace identifier: %s'%id_str)
    
    # return listing of all object 'info' in all workspaces user can read
    def _get_ws_list(self, wstype):
        return self._ws_post('list_objects', {'type': wstype, 'includeMetadata': 1})
    
    # returns obj with these fields: created, creator, info, provenance, data
    def _get_ws_object(self, ws_id):
        id_obj = self._parse_ws_id(ws_id)
        return self._ws_post('get_objects', [id_obj])[0]

    # delete obj
    def _delete_ws_obj(self, ws_id):
        id_obj = self._parse_ws_id(ws_id)
        self._ws_post('delete_objects', [id_obj], no_return=False)

    # returns 'info' of saveing existing obj
    def _save_ws_obj(self, ws_id, meta, data):
        id_obj = self._parse_ws_id(ws_id)
        save_obj = {
            'type': self.ws_type,
            'objid': id_obj['objid'],
            'meta': meta,
            'data': data
        }
        return self._ws_post('save_objects', {'id': id_obj['wsid'], [save_obj]})[0]
    
    # returns 'info' of saveing new obj
    def _new_ws_obj(self, name, ws, meta, data):
        save_obj = {
            'type': self.ws_type,
            'name': name,
            'meta': meta,
            'data': data
        }
        return self._ws_post('save_objects', {'workspace': ws, [save_obj]})[0]
    
    # returns 'info' of renamed obj
    def _rename_ws_obj(self, ws_id, name):
        id_obj = self._parse_ws_id(ws_id)
        return self._ws_post('rename_object', {'new_name': name, 'obj': id_obj})
    
    def _ws_post(self, method, params, no_return=False):
        arg_hash = {
            'method': 'Workspace.'+method,
            'params': [params],
            'version': '1.1',
            'id': str(random.random())[2:]
        }
        body = json.dumps(arg_hash, cls=JSONObjectEncoder)
        try:
            rpost = requests.post(url, headers={'Authorization': self.token}, data=body)
            rj = rpost.json()
        except Exception as e:
            raise web.HTTPError(504, u'Unable to connect to Workspace server %s: %s' %(url, e))
        if not (rpost.ok and rj and isinstance(rj, dict) and all([key in rj for key in ['id','result','error']])):
            raise web.HTTPError(500, u'Unable to POST to Workspace server %s: %s' %(url, rpost.raise_for_status()))
        if rj['error']:
            try:
                raise web.HTTPError(500, 'Workspace error: '+rj['error']['message'])
            except:
                raise web.HTTPError(500, 'Workspace error: '+rj['error'])
        if not no_return:
            return rj['result'][0]
