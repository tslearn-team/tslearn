// Inspired from scikit-learn's warning

$(function() {
    var msg = 'This is documentation for the unstable ' +
        'development version of tslearn. To use it, ' +
        '<a href="https://tslearn.readthedocs.io/en/stable/installation.html#using-latest-github-hosted-version">install the latest github-hosted version</a>. ' +
        'The latest stable ' +
        'release is <a href="https://tslearn.readthedocs.io/en/stable/">available there</a>.';
    $('.body[role=main]').prepend(
            '<div class="admonition warning alert alert-warning">' +
            '<p class="first admonition-title">Warning</p>' +
            '<p class="last">' + msg + '</p></div>');
});
