.. list-table::
   :widths: 35 25
   :header-rows: 1

   * - Dependency
     - Version specifiers

{% for dependency in dependencies %}
   * - {{ dependency.name }}
     - {{ dependency.specifier|string|replace(",", "\t") }}
{% endfor %}
