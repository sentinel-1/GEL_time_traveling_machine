#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://raw.githubusercontent.com/sentinel-1/GEL_time_traveling_machine/master/images/GEL_rocket_orange_light_alt_(1200x628).png" width="300" alt="Logo of the GEL Time Traveling Machine" />
# </center>

# # GEL (₾) დროში მოგზაურობის მანქანა
# 
# *მინიშნება: შეარჩიეთ სხვადასხვა თარიღები ქვემოთ მოცემულ ველებში და/ან შეცვალეთ თანხის ოდენობა და შედეგები განახლდება ავტომატურად.*
# 

# In[1]:


from IPython.display import display, HTML
display(HTML("""<div id="time-traveling-machine-container-ka"></div>"""))


# <br>
# <br>

# In[2]:


from datetime import datetime, timedelta
nb_st = datetime.utcnow()
print(f"\nNotebook START time: {nb_st} UTC\n")


# In[3]:


get_ipython().run_cell_magic('HTML', '', '<style>\n@media (max-width: 540px) {\n  .output .output_subarea {\n    max-width: 100%;\n  }\n}\n</style>\n<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'🔎 Show Python Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'⌦ Hide Python Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide();\n    $(\'div.input:contains("%%HTML")\').removeClass( "input")\n    $(\'div.input:contains("%%capture")\').removeClass("input")\n  });\n</script>\n<form action="javascript:code_toggle()">\n  <input type="submit" id="toggleButton" value="🔎 Show Python Code"\n         class="btn btn-default btn-lg">\n</form>\n')


# In[4]:


VERBOSE = False


# ## როგორ მუშაობს
# 
# წარსული თარიღებისთვის გამოიყენება საქსტატის სფი მონაცემები და გამოითვლება ლარის ღირებულების ზუსტი შესაბამისობა სხვადასხვა დროში, ხოლო სამომავლო თარიღების შემთხვევაში კი გამოითვლება ლარის ღირებულების მიახლოებითი (სავარაუდო) შესაბამისობა რაც მიღებულია წინასწარ შექმნილი სხვადასხვა მოდელებიდან შერჩეული ერთ-ერთი მოდელის გამოყენებით. აღნიშნული მოდელების დეტალური ტექნიკური აღწერა და შესაბამისი Python-კოდი სრულიად ღია სახით შეგიძლიათ იხილოთ ამავე დოკუმენტის [ინგლისურენოვან ვერსიაში](../).
# 

# გამოყენებული მონაცემები:
# - სფი (სამომხმარებლო ფასების ინდექსი) სტატისტიკური მონაცემები (Excel ფორმატში) და შესაბამისი მეტამონაცემები (PDF ფორმატში) მოპოვებულია 2022 წლის 21 ივნისს საქართველოს სტატისტიკის ეროვნული სამსახურის (საქსტატის) ვებსაიტიდან: [geostat.ge](https://www.geostat.ge/ka/)
# - იგივე მონაცემები გარკვეულ შემთხვევებში შესაძლოა ავტომატურ რეჟიმში იქნეს გადმოწერილი/განახლებული იგივე წყაროდან (ტექნიკური დეტალებისთვის იხილეთ Python კოდი ინგლისურენოვან ვერსიაში).

# In[5]:


with open("GEL_TTM_ka.html", "r") as f:
    raw_html = f.read()
    
with open("GEL_TTM_ka.js", "r") as f:
    raw_js = f.read()


if VERBOSE:
    print(raw_html)
    print(raw_js)


# In[6]:


##
# Send it back to the beggining of the notebook instead of showing here:
##
display(HTML("""<script type="application/javascript">
((fn)=>{
  if (document.readyState != 'loading'){
    fn();
} else {
    document.addEventListener('DOMContentLoaded', fn);
}
})(()=>{
let ttm_container = document.getElementById("time-traveling-machine-container-ka");
"""f"""
ttm_container.insertAdjacentHTML("afterbegin", `{raw_html}`);
let script = document.createElement('script');
script.type="application/javascript";
script.textContent = `"""
+ raw_js.replace("</script>", "<\/script>") 
+ """`;
ttm_container.parentNode.appendChild(script);
""""""
});
</script>
"""))


# In[7]:


print(f"\n ** Total Elapsed time: {datetime.utcnow() - nb_st} ** \n")
print(f"Notebook END time: {datetime.utcnow()} UTC\n")


# In[8]:


get_ipython().run_cell_magic('capture', '', '%mkdir OGP_classic_ka\n')


# In[9]:


get_ipython().run_cell_magic('capture', '', '%%file "OGP_classic_ka/conf.json"\n{\n  "base_template": "classic",\n  "preprocessors": {\n    "500-metadata": {\n      "type": "nbconvert.preprocessors.ClearMetadataPreprocessor",\n      "enabled": true,\n      "clear_notebook_metadata": true,\n      "clear_cell_metadata": true\n    },\n    "900-files": {\n      "type": "nbconvert.preprocessors.ExtractOutputPreprocessor",\n      "enabled": true\n    }\n  }\n}\n')


# In[10]:


get_ipython().run_cell_magic('capture', '', '%%file "OGP_classic_ka/index.html.j2"\n{%- extends \'classic/index.html.j2\' -%}\n{%- block html_head -%}\n\n{#  OGP attributes for shareability #}\n<meta property="og:url"          content="https://sentinel-1.github.io/GEL_time_traveling_machine/ka/" />\n<meta property="og:type"         content="article" />\n<meta property="og:title"        content="GEL (₾) დროში მოგზაურობის მანქანა" />\n<meta property="og:description"  content="რა გავლენა აქვს ინფლაციას თქვენს ჯიბეზე საქართველოში?" />\n<meta property="og:image"        content="https://raw.githubusercontent.com/sentinel-1/GEL_time_traveling_machine/master/images/GEL_rocket_orange_light_alt_(1200x628).png" />\n<meta property="og:image:alt"    content="Logo of the GEL Time Traveling Machine" />\n<meta property="og:image:type"   content="image/png" />\n<meta property="og:image:width"  content="1200" />\n<meta property="og:image:height" content="628" />\n    \n<meta property="article:published_time" content="2022-07-16T20:58:27+00:00" />\n<meta property="article:modified_time"  content="{{ resources.iso8610_datetime_utcnow }}" />\n<meta property="article:publisher"      content="https://sentinel-1.github.io" />\n<meta property="article:author"         content="https://github.com/sentinel-1" />\n<meta property="article:section"        content="datascience" />\n<meta property="article:tag"            content="datascience" />\n<meta property="article:tag"            content="Python" />\n<meta property="article:tag"            content="data" />\n<meta property="article:tag"            content="analytics" />\n<meta property="article:tag"            content="datavisualization" />\n<meta property="article:tag"            content="bigdataunit" />\n<meta property="article:tag"            content="visualization" />\n<meta property="article:tag"            content="inflation" />\n<meta property="article:tag"            content="GEL" />\n<meta property="article:tag"            content="Lari" />\n<meta property="article:tag"            content="CPI" />\n<meta property="article:tag"            content="timetravelingmachine" />\n    \n    \n{{ super() }}\n\n{%- endblock html_head -%}\n    \n    \n{% block body_header %}\n<body>\n    \n<div class="container">\n  <nav class="navbar navbar-default">\n    <div class="container-fluid">\n      <ul class="nav nav-pills  navbar-left">\n        <li role="presentation">\n          <a href="/ka/">\n            <svg xmlns="http://www.w3.org/2000/svg"\n                 viewBox="0 0 576 512" width="1em">\n              <path \n                fill="#999999"\nd="M 288,0 574,288 511,288 511,511 352,511 352,352 223,352 223,511 62,511 64,288 0,288 Z"\n              />\n            </svg> მთავარი\n          </a>\n        </li>\n      </ul>\n      <ul class="nav nav-pills  navbar-right">\n        <li role="presentation">\n          <a href="/GEL_time_traveling_machine/">🇬🇧 English </a>\n        </li>\n        <li role="presentation" class="active">\n          <a href="/GEL_time_traveling_machine/ka/">🇬🇪 ქართული</a>\n        </li>\n      </ul>\n    </div>\n  </nav>\n</div>\n\n\n\n  <div tabindex="-1" id="notebook" class="border-box-sizing">\n    <div class="container" id="notebook-container">    \n{% endblock body_header %}\n\n{% block body_footer %}\n    </div>\n  </div>\n  <footer>\n    <div class="container"\n         style="display:flex; flex-direction: row; justify-content: center; align-items: center;">\n      <p style="margin: 3.7em auto;"> © 2022\n        <a href="https://github.com/sentinel-1" target="_blank">Sentinel-1</a>\n      </p>\n      <!-- TOP.GE ASYNC COUNTER CODE -->\n      <div id="top-ge-counter-container" data-site-id="116052"\n           style="margin-right: 3.7em;float: right;"></div>\n      <script async src="//counter.top.ge/counter.js"></script>\n      <!-- / END OF TOP.GE COUNTER CODE -->\n      <!-- ANALYTICS.LAGOGAL.COM -->\n      <div id="analytics-lagogal-com-access" data-site-id="20221"\n           style="margin: 0;padding: 0;"></div>\n      <script async src="//analytics.lagogal.com/access.js"></script>\n      <!-- / END OF ANALYTICS.LAGOGAL.COM -->\n     </div>\n  </footer>\n</body>\n{% endblock body_footer %}\n')


# *მოცემული დოკუმენტი თავდაპირველად გამოქვეყნებულ იქნა Apache License (Version 2.0) ლიცენზიით შემდეგ GitHub რეპოზიტორზე: [sentinel-1/GEL_time_traveling_machine](https://github.com/sentinel-1/GEL_time_traveling_machine)*
# 
# მოცემულ დოკუმენტის ორიგინალ ვერსიასთან დაკავშირებულ საკითხებზე შესაბამისი უკუკავშირისათვის, რჩევებისათვის ან შენიშვნებისთვის (თუ რამეა) შეგიძლიათ ახალი Issue-ს შექმნის გზით დააყენოთ საკითხი მისივე GitHub რეპოზიტორის შესაბამის გვერდზე: [Issues page of the repository](https://github.com/sentinel-1/GEL_time_traveling_machine/issues)
