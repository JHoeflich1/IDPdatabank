"""Offline regression tests for repository downloads and AddData file metadata."""
import io
import json
from pathlib import Path
import tarfile
import zipfile
from unittest.mock import patch
from urllib.error import HTTPError

import libarchive
import pytest
import yaml

from DatabankLib import databankio as downloads


def zip_bytes(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


class Response(io.BytesIO):
    def __init__(self, content=b'', length=True, url='https://edmond.mpg.de/dataset.xhtml'):
        super().__init__(content)
        self.headers = {'Content-Length': str(len(content))} if length else {}
        self.url = url

    def geturl(self):
        return self.url


@pytest.mark.parametrize('extension,format', [('zip', 'zip'), ('tar', 'pax'),
                                             ('tar.gz', 'pax'), ('tgz', 'pax'), ('7z', '7zip')])
def test_archive_formats(tmp_path, extension, format):
    source = tmp_path / 'input.xtc'
    source.write_bytes(b'trajectory')
    archive = tmp_path / ('data.' + extension)
    options = {'filter_name': 'gzip'} if extension in ('tar.gz', 'tgz') else {}
    with libarchive.file_writer(str(archive), format, **options) as writer:
        writer.add_files(str(source), pathname='folder/input.xtc')
    result = downloads.extract_nested_file_from_archives(archive, 'folder/input.xtc', tmp_path / 'out')
    assert result.read_bytes() == b'trajectory'


def test_three_archive_layers_and_dot_prefix(tmp_path):
    content = zip_bytes({'./folder/run.xtc': b'trajectory'})
    content = zip_bytes({'dir/third.zip': content})
    archive = tmp_path / 'outer.zip'
    archive.write_bytes(zip_bytes({'second.zip': content}))
    result = downloads.extract_nested_file_from_archives(
        archive, 'second.zip/dir/third.zip/folder/run.xtc', tmp_path / 'out')
    assert result.read_bytes() == b'trajectory'


@pytest.mark.parametrize('path', ['/absolute.xtc', '../escape.xtc', 'a/../../escape', r'a\b', ''])
def test_reject_invalid_paths(path):
    with pytest.raises(ValueError):
        downloads.validate_source_path(path)


def test_missing_member_and_symlink(tmp_path):
    archive = tmp_path / 'archive.tar'
    with tarfile.open(archive, 'w') as writer:
        member = tarfile.TarInfo('link.xtc')
        member.type = tarfile.SYMTYPE
        member.linkname = '/etc/passwd'
        writer.addfile(member)
    with pytest.raises(FileNotFoundError):
        downloads.extract_file_from_archive(archive, 'absent', tmp_path / 'out')
    with pytest.raises(ValueError, match='regular file'):
        downloads.extract_file_from_archive(archive, 'link.xtc', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_download_replaces_truncated_cache_and_skips_complete(tmp_path):
    destination = tmp_path / 'run.xtc'
    destination.write_bytes(b'bad')
    with patch.object(downloads.urllib.request, 'urlopen', side_effect=lambda *a, **k: Response(b'complete')):
        assert downloads.download_resource_from_uri('https://example/file', destination) == 2
        assert destination.read_bytes() == b'complete'
        assert downloads.download_resource_from_uri('https://example/file', destination) == 1
        assert downloads.download_resource_from_uri('https://example/file', destination, True) == 2


def test_unknown_size_and_failed_download_are_atomic(tmp_path):
    destination = tmp_path / 'sub/run.xtc'
    with patch.object(downloads.urllib.request, 'urlopen', return_value=Response(b'complete', length=False)):
        downloads.download_resource_from_uri('https://example/file', destination)
    response = Response(b'short')
    response.headers['Content-Length'] = '100'
    with patch.object(downloads.urllib.request, 'urlopen', return_value=response):
        with pytest.raises(IOError, match='size mismatch'):
            downloads.download_resource_from_uri('https://example/file', destination, True)
    assert destination.read_bytes() == b'complete'
    assert list(destination.parent.iterdir()) == [destination]


def test_shared_archive_cached_and_missing_member_restored(tmp_path):
    content = zip_bytes({'folder/run.xtc': b'trajectory', 'folder/run.tpr': b'topology'})
    responses = []
    def open_url(*args, **kwargs):
        response = Response(content)
        responses.append(response)
        return response
    with patch.object(downloads.urllib.request, 'urlopen', side_effect=open_url):
        for filename in ['run.xtc', 'run.tpr', 'run.xtc']:
            downloads.download_resource_from_uri('https://example/archive.zip', tmp_path / filename,
                                                 source_path='archive.zip/folder/' + filename)
        assert (tmp_path / 'run.xtc').read_bytes() == b'trajectory'
        assert (tmp_path / 'run.tpr').read_bytes() == b'topology'
        (tmp_path / 'run.xtc').unlink()
        downloads.download_resource_from_uri('https://example/archive.zip', tmp_path / 'run.xtc',
                                             source_path='archive.zip/folder/run.xtc')
    assert (tmp_path / 'run.xtc').read_bytes() == b'trajectory'
    assert len(list((tmp_path / '.archives').rglob('archive.zip'))) == 1


def test_bare_archive_is_downloaded_without_extraction(tmp_path):
    content = zip_bytes({'run.xtc': b'x'})
    with patch.object(downloads.urllib.request, 'urlopen', return_value=Response(content)):
        downloads.download_resource_from_uri('https://example/a.zip', tmp_path / 'a.zip')
    assert (tmp_path / 'a.zip').read_bytes() == content


def test_file_normalization_adds_no_metadata():
    sim = {'DOI': '10.5281/zenodo.123', 'TRJ': [['outer.zip/dir/run.xtc']],
           'TPR': [['outer.zip/dir/run.tpr']], 'OTHER': [['do/not/modify']]}
    original_keys = set(sim)
    sources = downloads.prepare_file_sources(sim, ['TRJ', 'TPR'])
    assert sources == {'run.xtc': 'outer.zip/dir/run.xtc', 'run.tpr': 'outer.zip/dir/run.tpr'}
    assert sim['TRJ'] == [['run.xtc']]
    assert sim['OTHER'] == [['do/not/modify']]
    saved = yaml.safe_load(yaml.safe_dump(sim))
    assert set(saved) == original_keys
    assert saved['TRJ'] == [['run.xtc']]


def test_collision_rejected_before_mutation():
    sim = {'TRJ': [['one.zip/run.xtc'], ['two.zip/run.xtc']]}
    with pytest.raises(ValueError, match='share local filename'):
        downloads.prepare_file_sources(sim, ['TRJ'])
    assert sim['TRJ'][0][0] == 'one.zip/run.xtc'


def test_zenodo_archive_url():
    assert downloads.resolve_download_file_url(
        '10.5281/zenodo.123', 'archive.zip/folder/run.xtc', False
    ) == 'https://zenodo.org/record/123/files/archive.zip'


def test_dataverse_dataset_archive_resolution():
    def open_url(url, **kwargs):
        if '/api/info/version' in url:
            return Response(b'{"status":"OK"}')
        if '/api/access/datafile/:persistentId' in url:
            raise HTTPError(url, 404, 'dataset DOI', {}, io.BytesIO())
        if '/api/datasets/' in url:
            return Response(json.dumps({'data': {'latestVersion': {'files': [
                {'dataFile': {'filename': 'data.zip', 'id': 42}}]}}}).encode())
        return Response()
    with patch.object(downloads.urllib.request, 'urlopen', side_effect=open_url):
        assert downloads.resolve_download_file_url(
            '10.17617/3.ABC', 'data.zip/run.xtc'
        ) == 'https://edmond.mpg.de/api/access/datafile/42'


def test_dataverse_file_doi_archive_resolution():
    def open_url(url, **kwargs):
        if '/api/info/version' in url:
            return Response(b'{"status":"OK"}')
        return Response()
    with patch.object(downloads.urllib.request, 'urlopen', side_effect=open_url):
        uri = downloads.resolve_download_file_url('10.17617/3.ABC/file', 'data.zip/run.xtc')
    assert uri == 'https://edmond.mpg.de/api/access/datafile/:persistentId?persistentId=doi:10.17617/3.ABC/file'


@pytest.mark.parametrize('trajectory', ['data.zip/run.xtc', ['data.zip/run.xtc'],
                                      [['data.zip/run.xtc']], [['data.zip/run.xtc', 'sha1']]])
def test_adddata_yaml_file_formats(trajectory):
    from DatabankLib.databankLibrary import parse_valid_config_settings
    sim, files = parse_valid_config_settings({
        'SOFTWARE': 'GROMACS', 'DOI': '10.5281/zenodo.123', 'SYSTEM': 'test',
        'DIR_WRK': '/tmp', 'PREEQTIME': 0, 'TIMELEFTOUT': 0, 'COMPOSITION': {},
        'TRJ': trajectory, 'TPR': [['data.zip/run.tpr']]})
    assert files == ['data.zip/run.xtc', 'data.zip/run.tpr']
    assert sim['TRJ'][0][0] == 'data.zip/run.xtc'


def test_adddata_download_hash_save_and_local_analysis(tmp_path):
    """Exercise AddData downloads and hashing, then load the extracted trajectory."""
    import MDAnalysis as mda
    from DatabankLib import databankLibrary as library
    from DatabankLib.settings.engines import software_dict, get_struc_top_traj_fnames
    from DatabankLib.core import System

    # A real one-frame trajectory and matching structure, without external data.
    gro = tmp_path / 'run.gro'
    gro.write_text('test\n    1\n    1ALA     CA    1   0.100   0.200   0.300\n   1.0   1.0   1.0\n')
    universe = mda.Universe(str(gro))
    xtc = tmp_path / 'run.xtc'
    with mda.Writer(str(xtc), n_atoms=1) as writer:
        writer.write(universe.atoms)
    payloads = {'run.gro': gro.read_bytes(), 'run.xtc': xtc.read_bytes(), 'run.tpr': b'unsupported TPR'}
    archive = zip_bytes({'nested.zip': zip_bytes({'folder/' + k: v for k, v in payloads.items()})})
    config = {'SOFTWARE': 'GROMACS', 'DOI': '10.5281/zenodo.123', 'SYSTEM': 'test',
              'DIR_WRK': str(tmp_path), 'PREEQTIME': 0, 'TIMELEFTOUT': 0, 'COMPOSITION': {},
              'TRJ': 'outer.zip/nested.zip/folder/run.xtc',
              'TPR': [['outer.zip/nested.zip/folder/run.tpr']],
              'GRO': ['outer.zip/nested.zip/folder/run.gro']}
    parsed, _ = library.parse_valid_config_settings(config)
    sim = System(parsed)
    file_keys = [k for k, spec in software_dict['GROMACS'].items() if 'file' in spec.get('TYPE', '')]
    work = tmp_path / 'work'
    repository_archive = tmp_path / 'outer.zip'
    repository_archive.write_bytes(archive)
    with patch.object(downloads, 'resolve_download_file_url', return_value=repository_archive.as_uri()):
        local_files = downloads.download_simulation_files(sim, work, file_keys)
    for filename in local_files:
        assert (work / filename).read_bytes() == payloads[filename]
        assert len(library.calc_file_sha1_hash(work / filename)) == 40
    structure, topology, trajectory = get_struc_top_traj_fnames(sim, join_path=str(work))
    assert all(Path(p).is_file() for p in [structure, topology, trajectory])
    readme = tmp_path / 'README.yaml'
    readme.write_text(yaml.safe_dump(sim.readme))
    saved = yaml.safe_load(readme.read_text())
    assert set(saved) == set(config)
    assert saved['TRJ'] == [['run.xtc']]
    assert saved['TPR'] == [['run.tpr']]
    structure, _, trajectory = get_struc_top_traj_fnames(saved, join_path=str(work))
    restored = mda.Universe(structure, trajectory)
    assert len(restored.atoms) == 1
    assert len(restored.trajectory) == 1
    assert restored.atoms.positions[0] == pytest.approx([1, 2, 3])
    restored.trajectory.close()


def test_same_basename_different_archive_member_is_not_stale(tmp_path):
    archive = tmp_path / 'source.zip'
    archive.write_bytes(zip_bytes({'replica1/run.xtc': b'first', 'replica2/run.xtc': b'second'}))
    destination = tmp_path / 'work/run.xtc'
    for member, expected in [('replica1', b'first'), ('replica2', b'second')]:
        downloads.download_resource_from_uri(archive.as_uri(), destination,
                                             source_path=f'source.zip/{member}/run.xtc')
        assert destination.read_bytes() == expected
    destination.write_bytes(b'corrupt local extraction')
    downloads.download_resource_from_uri(archive.as_uri(), destination,
                                         source_path='source.zip/replica2/run.xtc')
    assert destination.read_bytes() == b'second'


def test_legacy_destination_and_uppercase_archive(tmp_path):
    archive = tmp_path / 'source.zip'
    archive.write_bytes(zip_bytes({'folder/run.xtc': b'trajectory'}))
    downloads.download_resource_from_uri(archive.as_uri(), tmp_path / 'work/archive.ZIP/folder/run.xtc')
    assert (tmp_path / 'work/run.xtc').read_bytes() == b'trajectory'


def test_optional_missing_file_and_ordinary_download(tmp_path):
    from DatabankLib.databankLibrary import parse_valid_config_settings
    parsed, _ = parse_valid_config_settings({
        'SOFTWARE': 'GROMACS', 'DOI': '10.5281/zenodo.123', 'SYSTEM': 'test',
        'DIR_WRK': str(tmp_path), 'PREEQTIME': 0, 'TIMELEFTOUT': 0, 'COMPOSITION': {},
        'TRJ': 'run.xtc', 'TPR': 'run.tpr', 'GRO': None})
    original = tmp_path / 'original'
    original.write_bytes(b'plain resource')
    with patch.object(downloads, 'resolve_download_file_url', return_value=original.as_uri()):
        downloads.download_simulation_files(parsed, tmp_path / 'work', ['TRJ', 'TPR'])
    assert (tmp_path / 'work/run.xtc').read_bytes() == b'plain resource'
