"""Lane administration controls are delivered through the launcher UI."""
from cryosparc_2d_projection.web import create_app
from tests.test_web import authenticate


def test_launcher_serves_multi_lane_and_local_capacity_controls(tmp_path):
    client = create_app(dict(data_dir=str(tmp_path), cryosparc_url='https://cryo.example',
                             public_url='http://localhost', allow_http=True),
                        authenticate=authenticate).test_client()
    markup = client.get('/').text
    assert 'id="lane-select"' in markup
    assert 'id="new-lane"' in markup and 'id="duplicate-lane"' in markup
    assert 'id="slurm-max_concurrent"' in markup
    assert 'id="slurm-template_path"' in markup
    assert 'id="preview-lane"' in markup and 'id="reload-template"' in markup
    assert 'id="lane-preview"' in markup
    assert client.get('/assets/lanes.js').status_code == 200


def test_lane_editor_is_outside_the_workflow_form(tmp_path):
    from html.parser import HTMLParser
    class FormOwner(HTMLParser):
        def __init__(self):
            super().__init__()
            self.form = None
            self.lane_form = None
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'form':
                self.form = attrs.get('id')
            if attrs.get('id') == 'slurm-cpus':
                self.lane_form = self.form
        def handle_endtag(self, tag):
            if tag == 'form':
                self.form = None
    page = FormOwner()
    page.feed((__import__('pathlib').Path(__file__).parents[1] / 'src/web_assets/index.html').read_text())
    assert page.lane_form != 'workflow-form'
