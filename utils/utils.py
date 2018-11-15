# Databricks notebook source
configure local timezone
spark.conf.set('spark.sql.session.timeZone', 'Singapore')

import os
from time import tzset
os.environ['TZ'] = 'Asia/Singapore'
tzset()

# COMMAND ----------

import os


class S3Bucket(object):
  """Class to wrap around a S3 bucket and mount at dbfs."""
  def __init__(self, bucketname, aws_access_key, aws_secret_key):
    self.name = bucketname
    self._aws_access_key = aws_access_key
    self._aws_secret_key = aws_secret_key
    self._is_mounted = False
    
  @property
  def is_mounted(self):
    """Whether the bucket is mounted."""
    return self._is_mounted

  def allowSpark(self):
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", self._aws_access_key)
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", self._aws_secret_key)
    return self
  
  def mount(self, mount_pt, ignore_exception=True):
    """
    Mounts the S3 bucket in dbfs.
    environment variables `AWS_ACCESS_KEY` and `AWS_SECRET_KEY` must be set.
    """
    self.mount_at = "/mnt/{0}".format(mount_pt)
    try:
      dbutils.fs.mount("s3a://%s:%s@%s" % (self._aws_access_key, self._aws_secret_key.replace("/", "%2F"), self.name), self.mount_at)
      display(dbutils.fs.ls("/mnt/%s" % mount_pt))
    except Exception as e:
      if "Directory already mounted" not in str(e):
        raise e
    self._is_mounted = True
    return self
    
  def umount(self):
    """umount the s3 bucket"""
    if self.is_mounted:
      dbutils.fs.unmount(self.mount_at)
      self._is_mounted = False
    return self
  
  def s3(self, path):
    """Return the path to the """
    return "s3a://" + self.name + "/" + path  
  
  def local(self, path):
    if self.is_mounted:
      return "/dbfs/{0}/{1}".format(self.mount_at, path)
    raise RuntimeError("Bucket is not mounted yet!")

# COMMAND ----------

def ls(path):
  """List the files and folders the s3 bucket"""
  files = dbutils.fs.ls(path)
  html = ""
  for file in files:
    classname = "file" if file.size > 0 else "folder"
    html += "<tr class='{0}'><td>{1}</td><td>{2}</td></tr>".format(classname, file.name, file.size)
    
  displayHTML("""
  <style>
  section {color: #666; font-family: sans-serif; font-size: 11px;}
  .folder {color: #1976d2; font-weight: bold;}
  .file {color: #00897b;}
  </style>
  <section>
    <big>"""+path+"""</big>
    <table>
      <tr><th>name</th><th>size</th></tr>
      """+html+"""
    </table>
  </section>
  """)
  return files

# COMMAND ----------

import json

def vega(spec, render=True):
  """Display a vega chart."""
  if isinstance(spec, dict):
    spec = json.dumps(spec)
    
  html = """
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/vega@3"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@2"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@3"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-themes@2"></script>
</head>
<body>

<div id="vis"></div>

<script type="text/javascript">
  var spec = """+spec+""";
  vegaEmbed('#vis', spec, {theme: 'quartz', defaultStyle: true, actions: {export: true, source: true, editor: false, renderer: 'svg'}}).catch(console.error);
</script>
</body>
</html>"""
  if render:
    displayHTML(html)
  return html
