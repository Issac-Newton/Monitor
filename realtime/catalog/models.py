from django.db import models
import uuid

# Create your models here.
from django.urls import reverse #Used to generate urls by reversing the URL patterns


class realtimeNodeInfo(models.Model):
	"""
	Model representing a book genre (e.g. Science Fiction, Non Fiction).
	"""
	id = models.BigAutoField(primary_key=True, default=uuid.uuid4, help_text="Unique ID for this system")
	runUser = models.IntegerField(help_text="the runUser of this system")
	idleNode = models.IntegerField(help_text = "the idleNode of this system")
	pendJob = models.IntegerField(help_text = "the pendJob of this system")
	nodeName = models.CharField(max_length=20, help_text=" the nodeName of this system")
	availableCore = models.IntegerField(help_text = "the availableCore of this system")
	nodeuprate = models.FloatField(help_text = "the nodeuprate of this system")
	offNode = models.IntegerField(help_text = "the offNode of this system")
	cpuutil = models.FloatField(help_text = "the cpuutil of this system")
	pendCore = models.IntegerField(help_text = "the pendCore of this system")
	allJob = models.IntegerField(help_text = "the allJob of this system")
	runJob = models.IntegerField(help_text = "the runJob of this system")
	occupyNode = models.IntegerField(help_text = "the occupyNode of this system")
	closedNode = models.IntegerField(help_text = "the closedNode of this system")
	allNode = models.IntegerField(help_text = "the allNode of this system")
	reserveNode = models.IntegerField(help_text = "the reserveNode of this system")
	nodeutil = models.FloatField(help_text = "the nodeutil of this system")
	usercount = models.IntegerField(help_text = "the usercount of this system")
	runCore = models.IntegerField(help_text = "the runCore of this system")
	penduser = models.IntegerField(help_text = "the penduser of this system")
	activeUser = models.IntegerField(help_text = "the activeUser of this system")

	def __str__(self):
		"""
		String for representing the Model object (in Admin site etc.)
		"""
		return self.nodeName
		
	def get_absolute_url(self):
		"""
		Returns the url to access a particular instance of the model.
		"""
		return reverse('realtimeNodeInfo', args=[str(self.nodeName)])

class cpuutilInfo(models.Model):
	id = models.BigAutoField(primary_key=True, default=uuid.uuid4, help_text="Unique ID for this system")
	casnw = models.FloatField(help_text="the cpuutil of casnw node")
	dicp = models.FloatField(help_text="the cpuutil of dicp node")
	era = models.FloatField(help_text="the cpuutil of era node")
	erai = models.FloatField(help_text="the cpuutil of erai node")
	gspcc = models.FloatField(help_text="the cpuutil of gspcc node")
	hku = models.FloatField(help_text="the cpuutil of hku node")
	hust = models.FloatField(help_text="the cpuutil of hust node")
	iapcm = models.FloatField(help_text="the cpuutil of iapcm node")
	nscccs = models.FloatField(help_text="the cpuutil of nscccs node")
	nsccgz = models.FloatField(help_text="the cpuutil of nsccgz node")
	nsccjn = models.FloatField(help_text="the cpuutil of nsccjn node")
	nscctj = models.FloatField(help_text="the cpuutil of nscctj node")
	nsccwx = models.FloatField(help_text="the cpuutil of jsccwx node")
	siat = models.FloatField(help_text="the cpuutil of siat node")
	sjtu = models.FloatField(help_text="the cpuutil of sjtu node")
	ssc = models.FloatField(help_text="the cpuutil of ssc node")
	ustc = models.FloatField(help_text="the cpuutil of ustc node")
	xjtu = models.FloatField(help_text="the cpuutil of xjtu node")
	anomaly = models.IntegerField(default = 0)

	def __str__(self):
		"""
		String for representing the Model object (in Admin site etc.)
		"""
		return self.id,self.casnw,self.anomaly

