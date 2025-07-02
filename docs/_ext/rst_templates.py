import os
import jinja2
import docutils.statemachine

CONFIG_VALUE = 'rst_templates'


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template with dedicated context.
    """
    src = source[0]
    relative_docname = os.path.relpath(docname, app.builder.srcdir)
    if not os.path.exists(docname):
        relative_docname = docname + '.rst'
    templates = getattr(app.config, CONFIG_VALUE, {})
    if relative_docname in templates:
        rendered = jinja2.Template(src).render(
            **templates.get(relative_docname, {})
        )
        source[0] = rendered


def setup(app):
    """ Rst templating """
    # Because include does not trigger source-read on sphinx 7.1
    og_insert_input = docutils.statemachine.StateMachine.insert_input
    def my_insert_input(self, include_lines, path):
        # first we need to combine the lines back into text so we can send it with the source-read
        # event:
        text = "\n".join(include_lines)
        # emit "source-read" event
        arg = [text]
        app.env.events.emit("source-read", path, arg)
        text = arg[0]
        # split into lines again:
        include_lines = text.splitlines()
        # call the original function:
        og_insert_input(self, include_lines, path)
    # inject our patched function
    docutils.statemachine.StateMachine.insert_input = my_insert_input

    # Add
    app.add_config_value(CONFIG_VALUE, {}, True, dict)
    app.connect("source-read", rstjinja)
