# Doccano: Text Annotation Tool

## What is Doccano?

[Doccano](https://github.com/chakki-works/doccano) is one of the best open source tools that provides text annotation features. The latest version supports annotation features for text classification, sequence labeling (NER) and sequence to sequence (machine translation, text summarization). There are many other open source and commercial annotation tools available. Hereafter is an list of those tools:

- [Brat](https://brat.nlplab.org/) (open source)
- [Anafora](https://github.com/weitechen/anafora) (open source)
- [Prodigy](https://prodi.gy/) (commercial)
- [LightTag](https://www.lighttag.io/) (commercial)

Doccano needs to be hosted somewhere such that we can collaborate it. This tutorial walks through how to deploy Doccano on Azure and collaboratively annotate text data for natural language processing tasks.

## Deploy to Azure

Doccano can be deployed to Azure ([Web App for Containers](https://azure.microsoft.com/en-us/services/app-service/containers/) +
[PostgreSQL database](https://azure.microsoft.com/en-us/services/postgresql/)) by clicking on the button below:

<p align="center">
  <a href="https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fchakki-works%2Fdoccano%2Fmaster%2Fazuredeploy.json"><img width=180 src="https://nlpbp.blob.core.windows.net/images/deploybutton.jpg" /></a>
</p>

You will need to have an existing Azure subscription such that you can create all Azure resources need to deploy Doccano. Otherwise you can get a [free Azure account](https://azure.microsoft.com/en-us/offers/ms-azr-0044p/?WT.mc_id=medium-blog-abornst) and then click the deploy button above.

You will need to specify your subscription and resource group, and fill in the setting details (App Name, Secret Key, and etc.) and then deploy. It takes a few minutes to create all needed Azure resources. Hereafter is a screen snippet of the deployment. 

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/deploy_to_azure.jpg" />
</p>

## Tutorial

### Useful Links

#### Main Page

After the deployment you can navigate to following url where **{`appname`}** is the `App Name` you choose when deploy to Azure:

_**https://{appname}.azurewebsites.net**_

For example, if your appname is "**doccano**", then the link will be

_**https://doccano.azurewebsites.net**_

And we will use `doccano` as the app name for this tutorial.

#### Login Page

You can login by clicking the `login` button at the top right of the main page, or you can navigate to the page with the link

_**https://doccano.azurewebsites.net/login**_

Both will bring you in to the Doccano login page where you can login with the Admin user name and Admin password you configured in the deployment. 

#### Admin Page

By default, only the Admin user is created for you after the deployment. You can add more users, groups and configure the Doccano service by navigating to the admin page.

_**https://doccano.azurewebsites.net/admin**_

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/admin_page.JPG" />
</p>

### Create Project

The first step we need to do is to create a new project for annotation. And here we will use the NER annotation task for science fictions to give you a brief tutorial on Doccano. 

After login with Admin user name and Admin password, you will be navigated to the main project list page of Doccano and there is no project. 

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/project_list.jpg" />
</p>

To create your project, make sure you’re in the project list page and click `Create Project` button. As for this tutorial, we name the project as `sequence labeling for books`, write some description, then choose the sequence labeling task type.

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/create_project.jpg" />
</p>

### Import Data

After creating a project, we will see the "`Import Data`" page, or click `Import Data` button in the navigation bar. We should see the following screen:
<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/import_data.jpg" />
</p>

We choose JSONL and click `Select a file` button. Select `books.json` and it would be loaded automatically. Below is the `books.json` file containing lots of science fictions description with different languages. We need to annotate some entities like people name, book title, date and so on. 

```json
{"text": "The Hitchhiker's Guide to the Galaxy (sometimes referred to as HG2G, HHGTTGor H2G2) is a comedy science fiction series created by Douglas Adams. Originally a radio comedy broadcast on BBC Radio 4 in 1978, it was later adapted to other formats, including stage shows, novels, comic books, a 1981 TV series, a 1984 video game, and 2005 feature film."}
{"text": "《三体》是中国大陆作家刘慈欣于2006年5月至12月在《科幻世界》杂志上连载的一部长篇科幻小说，出版后成为中国大陆最畅销的科幻长篇小说之一。2008年，该书的单行本由重庆出版社出版。本书是三体系列（系列原名为：地球往事三部曲）的第一部，该系列的第二部《三体II：黑暗森林》已经于2008年5月出版。2010年11月，第三部《三体III：死神永生》出版发行。 2011年，“地球往事三部曲”在台湾陆续出版。小说的英文版获得美国科幻奇幻作家协会2014年度“星云奖”提名，并荣获2015年雨果奖最佳小说奖。"}
{"text": "『銀河英雄伝説』（ぎんがえいゆうでんせつ）は、田中芳樹によるSF小説。また、これを原作とするアニメ、漫画、コンピュータゲーム、朗読、オーディオブック等の関連作品。略称は『銀英伝』（ぎんえいでん）。原作は累計発行部数が1500万部を超えるベストセラー小説である。1982年から2009年6月までに複数の版で刊行され、発行部数を伸ばし続けている。"}
```

After importing the dataset, you should be able to see the dataset immediately. 

### Define labels

Click `Labels` button in left bar to define our own labels. We should see the label editor page. In label editor page, you can create labels by specifying label text, shortcut key, background color and text color.

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/define_labels.jpg" />
</p>

### Annotation

Next, we are ready to annotate the texts. Just click the `Annotate Data` button in the navigation bar, we can start to annotate the documents. You can just select the text and then use the shortcut key that you have defined to label the entities. 

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/annotate.jpg" />
</p>

### Export Data

After the annotation step, we can download the annotated data. Click the `Edit data` button in the navigation bar, and then click `Export Data`. You should see below screen:

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/export_data.jpg" />
</p>

Here we choose JSONL file to download the data by clicking the button. Below is the annotated result for our tutorial project.

```json
{"id": 1, "text": "The Hitchhiker's Guide to the Galaxy (sometimes referred to as HG2G, HHGTTGor H2G2) is a comedy science fiction series created by Douglas Adams. Originally a radio comedy broadcast on BBC Radio 4 in 1978, it was later adapted to other formats, including stage shows, novels, comic books, a 1981 TV series, a 1984 video game, and 2005 feature film.", "annotations": [{"label": 2, "start_offset": 0, "end_offset": 36, "user": 1}, {"label": 2, "start_offset": 63, "end_offset": 67, "user": 1}, {"label": 2, "start_offset": 69, "end_offset": 82, "user": 1}, {"label": 5, "start_offset": 89, "end_offset": 111, "user": 1}, {"label": 1, "start_offset": 130, "end_offset": 143, "user": 1}, {"label": 5, "start_offset": 158, "end_offset": 180, "user": 1}, {"label": 6, "start_offset": 184, "end_offset": 195, "user": 1}, {"label": 3, "start_offset": 199, "end_offset": 203, "user": 1}, {"label": 5, "start_offset": 254, "end_offset": 265, "user": 1}, {"label": 5, "start_offset": 267, "end_offset": 273, "user": 1}, {"label": 5, "start_offset": 275, "end_offset": 286, "user": 1}, {"label": 3, "start_offset": 290, "end_offset": 294, "user": 1}, {"label": 5, "start_offset": 295, "end_offset": 304, "user": 1}, {"label": 3, "start_offset": 308, "end_offset": 312, "user": 1}, {"label": 5, "start_offset": 313, "end_offset": 323, "user": 1}, {"label": 3, "start_offset": 329, "end_offset": 333, "user": 1}, {"label": 5, "start_offset": 334, "end_offset": 346, "user": 1}], "meta": {}, "annotation_approver": "admin"}
{"id": 2, "text": "《三体》是中国大陆作家刘慈欣于2006年5月至12月在《科幻世界》杂志上连载的一部长篇科幻小说，出版后成为中国大陆最畅销的科幻长篇小说之一。2008年，该书的单行本由重庆出版社出版。本书是三体系列（系列原名为：地球往事三部曲）的第一部，该系列的第二部《三体II：黑暗森林》已经于2008年5月出版。2010年11月，第三部《三体III：死神永生》出版发行。 2011年，“地球往事三部曲”在台湾陆续出版。小说的英文版获得美国科幻奇幻作家协会2014年度“星云奖”提名，并荣获2015年雨果奖最佳小说奖。", "annotations": [{"label": 2, "start_offset": 1, "end_offset": 3, "user": 1}, {"label": 4, "start_offset": 5, "end_offset": 9, "user": 1}, {"label": 1, "start_offset": 11, "end_offset": 14, "user": 1}, {"label": 3, "start_offset": 15, "end_offset": 26, "user": 1}, {"label": 2, "start_offset": 28, "end_offset": 32, "user": 1}, {"label": 5, "start_offset": 41, "end_offset": 47, "user": 1}, {"label": 4, "start_offset": 53, "end_offset": 57, "user": 1}, {"label": 5, "start_offset": 61, "end_offset": 67, "user": 1}, {"label": 3, "start_offset": 70, "end_offset": 74, "user": 1}, {"label": 6, "start_offset": 83, "end_offset": 88, "user": 1}, {"label": 2, "start_offset": 105, "end_offset": 112, "user": 1}, {"label": 2, "start_offset": 94, "end_offset": 98, "user": 1}, {"label": 2, "start_offset": 126, "end_offset": 135, "user": 1}, {"label": 3, "start_offset": 139, "end_offset": 146, "user": 1}, {"label": 3, "start_offset": 149, "end_offset": 157, "user": 1}, {"label": 2, "start_offset": 162, "end_offset": 172, "user": 1}, {"label": 3, "start_offset": 179, "end_offset": 184, "user": 1}, {"label": 2, "start_offset": 186, "end_offset": 193, "user": 1}, {"label": 4, "start_offset": 195, "end_offset": 197, "user": 1}, {"label": 5, "start_offset": 202, "end_offset": 204, "user": 1}, {"label": 6, "start_offset": 210, "end_offset": 220, "user": 1}, {"label": 3, "start_offset": 220, "end_offset": 225, "user": 1}, {"label": 6, "start_offset": 227, "end_offset": 230, "user": 1}, {"label": 3, "start_offset": 237, "end_offset": 242, "user": 1}, {"label": 6, "start_offset": 242, "end_offset": 245, "user": 1}], "meta": {}, "annotation_approver": "admin"}
{"id": 3, "text": "『銀河英雄伝説』（ぎんがえいゆうでんせつ）は、田中芳樹によるSF小説。また、これを原作とするアニメ、漫画、コンピュータゲーム、朗読、オーディオブック等の関連作品。略称は『銀英伝』（ぎんえいでん）。原作は累計発行部数が1500万部を超えるベストセラー小説である。1982年から2009年6月までに複数の版で刊行され、発行部数を伸ばし続けている。", "annotations": [{"label": 2, "start_offset": 1, "end_offset": 7, "user": 1}, {"label": 1, "start_offset": 23, "end_offset": 30, "user": 1}, {"label": 5, "start_offset": 30, "end_offset": 34, "user": 1}, {"label": 2, "start_offset": 85, "end_offset": 88, "user": 1}, {"label": 5, "start_offset": 50, "end_offset": 52, "user": 1}, {"label": 5, "start_offset": 63, "end_offset": 65, "user": 1}, {"label": 3, "start_offset": 130, "end_offset": 135, "user": 1}, {"label": 3, "start_offset": 137, "end_offset": 144, "user": 1}], "meta": {}, "annotation_approver": "admin"}
```

Please note that in the exported JSON file, the label for each entity is an entity ID which is inconvenient if you want to consume the annotations somewhere else. Some post processing is needed if you want to have the entity type value instead of the type ID.

### View Statistics

One good thing of Doccano is that it also has dashboard to display annotation progress and label distributions. Click the `Edit data` button in the navigation bar, and then click `Statistics` on the left side of the menu.

<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/statistic.jpg" />
</p>

Congratulation! You just mastered how to use Doccano for a sequence labeling project.
