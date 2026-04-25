{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# All Required Libraries"
      ],
      "metadata": {
        "id": "xzDkXZK0KReB"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9GjKFx5PKLaV",
        "outputId": "76b0ebde-f163-44ee-c118-1c5d87792493"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n"
          ]
        }
      ],
      "source": [
        "# Core Libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Text Processing\n",
        "import re\n",
        "import string\n",
        "\n",
        "# NLP\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "\n",
        "nltk.download('stopwords')\n",
        "\n",
        "# ML\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "\n",
        "# Models\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "# Evaluation\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix\n",
        "\n",
        "# Visualization\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Upload Dataset"
      ],
      "metadata": {
        "id": "FbcNr5r1KZVn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 106
        },
        "id": "xnGScfSrKcr2",
        "outputId": "d5dc5ce7-651a-44f3-bbfa-ee1c433225a4"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-ea40b7aa-e983-46d0-ab8c-4b082b88f48d\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-ea40b7aa-e983-46d0-ab8c-4b082b88f48d\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving Fake.csv to Fake.csv\n",
            "Saving True.csv to True.csv\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Load the Uploaded Dataset"
      ],
      "metadata": {
        "id": "jlabdcPLKglH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load datasets\n",
        "fake_df = pd.read_csv('Fake.csv')\n",
        "true_df = pd.read_csv('True.csv')\n",
        "\n",
        "# Add labels\n",
        "fake_df['label'] = 0\n",
        "true_df['label'] = 1\n",
        "\n",
        "# Combine datasets\n",
        "df = pd.concat([fake_df, true_df], axis=0)\n",
        "\n",
        "# Shuffle dataset\n",
        "df = df.sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "\n",
        "print(\"Dataset Shape:\", df.shape)\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 224
        },
        "id": "RRaAMRzdKihX",
        "outputId": "fe3f1826-1fda-4de4-eea4-ca916400b10e"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset Shape: (44898, 5)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                               title  \\\n",
              "0  Ben Stein Calls Out 9th Circuit Court: Committ...   \n",
              "1  Trump drops Steve Bannon from National Securit...   \n",
              "2  Puerto Rico expects U.S. to lift Jones Act shi...   \n",
              "3   OOPS: Trump Just Accidentally Confirmed He Le...   \n",
              "4  Donald Trump heads for Scotland to reopen a go...   \n",
              "\n",
              "                                                text       subject  \\\n",
              "0  21st Century Wire says Ben Stein, reputable pr...       US_News   \n",
              "1  WASHINGTON (Reuters) - U.S. President Donald T...  politicsNews   \n",
              "2  (Reuters) - Puerto Rico Governor Ricardo Rosse...  politicsNews   \n",
              "3  On Monday, Donald Trump once again embarrassed...          News   \n",
              "4  GLASGOW, Scotland (Reuters) - Most U.S. presid...  politicsNews   \n",
              "\n",
              "                  date  label  \n",
              "0    February 13, 2017      0  \n",
              "1       April 5, 2017       1  \n",
              "2  September 27, 2017       1  \n",
              "3         May 22, 2017      0  \n",
              "4       June 24, 2016       1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e8ee834c-816a-47f7-9990-25a7a17900f4\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>title</th>\n",
              "      <th>text</th>\n",
              "      <th>subject</th>\n",
              "      <th>date</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Ben Stein Calls Out 9th Circuit Court: Committ...</td>\n",
              "      <td>21st Century Wire says Ben Stein, reputable pr...</td>\n",
              "      <td>US_News</td>\n",
              "      <td>February 13, 2017</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Trump drops Steve Bannon from National Securit...</td>\n",
              "      <td>WASHINGTON (Reuters) - U.S. President Donald T...</td>\n",
              "      <td>politicsNews</td>\n",
              "      <td>April 5, 2017</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Puerto Rico expects U.S. to lift Jones Act shi...</td>\n",
              "      <td>(Reuters) - Puerto Rico Governor Ricardo Rosse...</td>\n",
              "      <td>politicsNews</td>\n",
              "      <td>September 27, 2017</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>OOPS: Trump Just Accidentally Confirmed He Le...</td>\n",
              "      <td>On Monday, Donald Trump once again embarrassed...</td>\n",
              "      <td>News</td>\n",
              "      <td>May 22, 2017</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Donald Trump heads for Scotland to reopen a go...</td>\n",
              "      <td>GLASGOW, Scotland (Reuters) - Most U.S. presid...</td>\n",
              "      <td>politicsNews</td>\n",
              "      <td>June 24, 2016</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e8ee834c-816a-47f7-9990-25a7a17900f4')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e8ee834c-816a-47f7-9990-25a7a17900f4 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e8ee834c-816a-47f7-9990-25a7a17900f4');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 44898,\n  \"fields\": [\n    {\n      \"column\": \"title\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 38729,\n        \"samples\": [\n          \" WATCH: Dem Senator BLASTS Trump, Calls Him A Liar Live On Air\",\n          \"Trump calls for firm response to North Korea, targets Seoul on trade\",\n          \"Zimbabwe army leaves streets a month after Mugabe's ouster\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"text\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 38646,\n        \"samples\": [\n          \"It takes one to know one. Turkey just held a referendum that greatly expands the power of their president, Recep Tayyip Erdogan. It passed by a very narrow margin, taking Turkey on its latest step toward brutal dictatorship, and here s Donald Trump, who sources say called Erdogan to congratulate him on  winning  the referendum vote.While we re busy justifying blowing up absolutely nothing in Syria because a brutal dictator used sarin gas on his people, Trump is busy calling someone who s working hard on becoming the region s next brutal dictator to congratulate him on furthering that goal.This referendum, according to The Daily Beast, moves Turkey away from a parliamentary democracy and towards one-person rule. But what he has already done there makes the referendum more of a formality. Erdogan had already managed to form a one-party government   a move that greatly diminishes the voices of opposition.Last year, Erdogan asked Turkey s parliament to redefine the country s anti-extremism law to include politicians, journalists and members of academia. He claimed that  pro-Kurdish  politicians were inciting terrorism, and journalists and academics were spreading the info that allowed the politicians to do so. Therefore, they are all terrorists.Branding press as  the enemy  is something Trump has been trying to do here. As the Washington Post s front page motto says,  Democracy dies in darkness.  This is the darkness.And now, Erdogan is, more or less, the sole ruler of Turkey.But what does Trump care? It wouldn t be surprising to find that he wishes something like that would happen here, too, if for no other reason than it would help cement his overinflated opinion of himself as a great man who is beloved by all, with nobody left to shine a light on the truth, like, oh, say, a free press.The way the Turkey referendum was held has appalled international election monitors. According to them,  voters were not provided with adequate information, opposition voices were muzzled and the rules were changed at the last minute.  In short, this was not a truly democratic process.Good job continuing to support authoritarian rulers over true democracies, Trump. You re about as un-American as it gets.Featured image by Mark Wilson via Getty Images\",\n          \"Donald Trump just got caught lying again and there is video to prove it.When former FBI Director James Comey testified under oath in the Senate on Thursday, he recalled that Trump demanded loyalty from him before he was asked to drop the investigation of Michael Flynn and Trump s ties to Russia. The President said,  I need loyalty, I expect loyalty,  Comey testified.But when asked by ABC reporter Jon Karl if he had demanded Comey s loyalty, Trump denied the whole thing and pretended that he has never demanded loyalty from anyone in his entire life. So he said those things under oath,  Karl began.  Would you be willing to speak under oath to give your version of those events? One hundred percent,  Trump replied.  I didn t say under oath   I hardly know the man. I m not going to say, I want you to pledge allegiance. Who would do that? Who would ask a man to pledge allegiance under oath? I mean, think of it. I hardly know the man. It doesn t make sense. No, I didn t say that, and I didn t say the other. If Trump were to say this under oath he would he would be committing perjury. And CNN proved it by playing video of Trump on the campaign trail in Florida asking people in the crowd to raise their hands and pledge their loyalty to him. Keep in mind that Trump had never met anyone in that audience. But he had several discussions and meetings with Comey. In other words, Comey was not a complete stranger to him. If he was willing to ask a crowd of random people to pledge their allegiance to him, he would certainly be willing to demand loyalty from Comey.Here s the damning video via YouTube.This is yet more proof that Trump is a liar who has zero credibility. He will say anything to save his own ass.Featured image via Olivier Douliery   Pool/Getty Images\",\n          \"WASHINGTON (Reuters) - The U.S. State Department said on Monday Washington was  very concerned  by reports of violence around the Iraqi oil city of Kirkuk, which was seized by Baghdad s forces from Kurds.  We are monitoring the situation closely and call on all parties to coordinate military activities and restore calm,   State Department spokeswoman Heather Nauert said in a statement. \"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"subject\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"politicsNews\",\n          \"worldnews\",\n          \"US_News\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"date\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2397,\n        \"samples\": [\n          \"July 3, 2016\",\n          \"Jul 29, 2015\",\n          \"Nov 28, 2016\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"label\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Cleaning & Preprocessing Function"
      ],
      "metadata": {
        "id": "Ss1qhU7mKpOv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "def preprocess_text(text):\n",
        "    text = str(text).lower()\n",
        "\n",
        "    # Remove URLs\n",
        "    text = re.sub(r'http\\S+|www\\S+', '', text)\n",
        "\n",
        "    # Remove HTML tags\n",
        "    text = re.sub(r'<.*?>', '', text)\n",
        "\n",
        "    # Remove punctuation\n",
        "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
        "\n",
        "    # Remove numbers\n",
        "    text = re.sub(r'\\d+', '', text)\n",
        "\n",
        "    # Tokenization\n",
        "    tokens = text.split()\n",
        "\n",
        "    # Remove stopwords\n",
        "    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]\n",
        "\n",
        "    return \" \".join(tokens)"
      ],
      "metadata": {
        "id": "eeonVkfnKqYP"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Apply Preprocessing"
      ],
      "metadata": {
        "id": "S8VDcFN3KsCA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Combine title + text\n",
        "df['content'] = df['title'] + \" \" + df['text']\n",
        "\n",
        "# Apply cleaning\n",
        "df['content'] = df['content'].apply(preprocess_text)\n",
        "\n",
        "print(df['content'].head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WEDNcbVZKtIH",
        "outputId": "5410f63b-8818-461f-ec7e-9aca4ce45e99"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0    ben stein calls circuit court committed ‘coup ...\n",
            "1    trump drops steve bannon national security cou...\n",
            "2    puerto rico expects lift jones act shipping re...\n",
            "3    oops trump accidentally confirmed leaked israe...\n",
            "4    donald trump heads scotland reopen golf resort...\n",
            "Name: content, dtype: object\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature Engineering (TF-IDF – Optimized)"
      ],
      "metadata": {
        "id": "32yiMlGYKw4K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Remove strong source-bias words BEFORE vectorization\n",
        "bias_words = ['reuters', 'said', 'washington', 'tuesday', 'wednesday']\n",
        "\n",
        "def remove_bias_words(text):\n",
        "    for word in bias_words:\n",
        "        text = text.replace(word, '')\n",
        "    return text\n",
        "\n",
        "df['content'] = df['content'].apply(remove_bias_words)\n",
        "\n",
        "# Improved TF-IDF\n",
        "tfidf = TfidfVectorizer(\n",
        "    max_features=8000,\n",
        "    ngram_range=(1,2),\n",
        "    min_df=5,          # ignore rare words\n",
        "    max_df=0.7,        # ignore overly common words\n",
        "    stop_words='english'\n",
        ")\n",
        "\n",
        "X = tfidf.fit_transform(df['content'])\n",
        "y = df['label']\n",
        "\n",
        "print(\"Feature matrix shape:\", X.shape)"
      ],
      "metadata": {
        "id": "n_r2Ob34KxVD",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "59b5e659-cdac-4fd0-e976-7113a2395fa6"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Feature matrix shape: (44898, 8000)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Train-Test Split"
      ],
      "metadata": {
        "id": "Vw4s8cwcKzai"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y,\n",
        "    test_size=0.25,   # slightly larger test set\n",
        "    random_state=42,\n",
        "    stratify=y\n",
        ")"
      ],
      "metadata": {
        "id": "ANvdHnswK0Rq"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Train Models"
      ],
      "metadata": {
        "id": "IdER7DIuK1W3"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Naive Bayes"
      ],
      "metadata": {
        "id": "SqVGLNlvK3YW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "nb_model = MultinomialNB()\n",
        "nb_model.fit(X_train, y_train)\n",
        "\n",
        "nb_pred = nb_model.predict(X_test)"
      ],
      "metadata": {
        "id": "lxITNhWvK38K"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Logistic Regression"
      ],
      "metadata": {
        "id": "Z6vW46TeK5Ii"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "lr_model = LogisticRegression(\n",
        "    max_iter=1000,\n",
        "    class_weight='balanced'\n",
        ")\n",
        "\n",
        "param_grid = {\n",
        "    'C': [0.1, 1, 5, 10]\n",
        "}\n",
        "\n",
        "grid = GridSearchCV(lr_model, param_grid, cv=5, scoring='f1')\n",
        "grid.fit(X_train, y_train)\n",
        "\n",
        "best_lr = grid.best_estimator_\n",
        "\n",
        "print(\"Best Parameters:\", grid.best_params_)\n",
        "\n",
        "lr_pred = best_lr.predict(X_test)"
      ],
      "metadata": {
        "id": "x4l13UybK7hG",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9ed39d97-1d1a-4586-c0c7-88cadabe7bee"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best Parameters: {'C': 10}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluation Function"
      ],
      "metadata": {
        "id": "Ia8pBUz0K9Hl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_model(y_test, y_pred, model_name):\n",
        "    print(f\"\\n===== {model_name} =====\")\n",
        "\n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    prec = precision_score(y_test, y_pred)\n",
        "    rec = recall_score(y_test, y_pred)\n",
        "    f1 = f1_score(y_test, y_pred)\n",
        "\n",
        "    print(f\"Accuracy:  {acc:.4f}\")\n",
        "    print(f\"Precision: {prec:.4f}\")\n",
        "    print(f\"Recall:    {rec:.4f}\")\n",
        "    print(f\"F1 Score:  {f1:.4f}\")\n",
        "\n",
        "    print(\"\\nClassification Report:\\n\")\n",
        "    print(classification_report(y_test, y_pred))\n",
        "\n",
        "    # Confusion Matrix\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "\n",
        "    plt.figure()\n",
        "    sns.heatmap(cm, annot=True, fmt='d')\n",
        "    plt.title(f\"{model_name} - Confusion Matrix\")\n",
        "    plt.xlabel(\"Predicted\")\n",
        "    plt.ylabel(\"Actual\")\n",
        "    plt.show()\n",
        "\n",
        "    return acc, prec, rec, f1"
      ],
      "metadata": {
        "id": "Eh8rtk4dK-KG"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluate Models"
      ],
      "metadata": {
        "id": "DNNki9YcK_Sx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "nb_results = evaluate_model(y_test, nb_pred, \"Naive Bayes\")\n",
        "lr_results = evaluate_model(y_test, lr_pred, \"Logistic Regression\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "T1LYRJL9LAWX",
        "outputId": "76477ea6-071c-406c-c6fd-4c480b3e0603"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "===== Naive Bayes =====\n",
            "Accuracy:  0.9352\n",
            "Precision: 0.9368\n",
            "Recall:    0.9268\n",
            "F1 Score:  0.9317\n",
            "\n",
            "Classification Report:\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.93      0.94      0.94      5871\n",
            "           1       0.94      0.93      0.93      5354\n",
            "\n",
            "    accuracy                           0.94     11225\n",
            "   macro avg       0.94      0.93      0.94     11225\n",
            "weighted avg       0.94      0.94      0.94     11225\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiQAAAHHCAYAAACPy0PBAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAASzxJREFUeJzt3XlcVdX6x/HvAeWIICjKmBOpqZhZaSmZU5JYaJqa2eDcoKGlllOD2khaXYdMrczwllZqN38qqZFjJVlh5JBDjjiBogJODML+/eHlXI+oB+xsD+Hn3Wu/XrL2OmuvTRx9eJ619rEYhmEIAADAhdxcPQEAAAACEgAA4HIEJAAAwOUISAAAgMsRkAAAAJcjIAEAAC5HQAIAAFyOgAQAALgcAQkAAHA5AhI4RevWrdW6dWtXTwMudO7cOY0YMULVqlWTm5ubOnfu7PRr8HNmLzY2VhaLRXv37nX1VIC/jYDkOlLwl1e5cuV08ODBQudbt26tm2++2QUzu3qtW7eWxWKxHR4eHgoNDdVTTz2l/fv3u3p611xWVpYmTpyopk2bytfXV+XKldNNN92kQYMGaceOHaZee9asWXrnnXfUrVs3zZ49W0OHDjX1etfS6tWrbT9jn3/++SX7NG/eXBaL5arfQ9OmTVNsbOzfmCXwz2bhs2yuH7Gxserbt68kadCgQXr//fftzrdu3VppaWnavHlzscfOycmRJHl4ePz9iRZD69attWvXLsXExNjm8eeff2rGjBmqXLmytm7dqvLly1/TOblKWlqa2rdvr8TERHXo0EERERHy9vbW9u3b9eWXXyolJcX2/8kMPXr00I8//qgDBw6Ydg1X/ZytXr1abdq0Ubly5dSmTRt9++23duf37t2r0NBQlStXTrVq1bqq99DNN9+sKlWqaPXq1UV+TV5ennJzc2W1WmWxWIp9TaAkKePqCeDau/XWW/Xxxx9r9OjRCgkJccqY1/ofiAv5+vrq8ccft2sLDQ3VoEGD9NNPP+nee+910cyurT59+uj333/XggUL1LVrV7tzr7/+ul566SVTr3/kyBFVrFjR1Gu48udMku6//34tWrRIaWlpqlKliq197ty5CgwMVJ06dXTixAnT53H69Gl5eXnJ3d1d7u7upl8PuBYo2VyHXnzxReXl5entt9922PfTTz/VPffco4CAAFmtVoWFhWn69OmF+l1Y209NTVWZMmX06quvFuq3fft2WSwWTZ061daWnp6uIUOGqFq1arJarapdu7bGjx+v/Pz8q77HoKAgSVKZMv+Lufft26dnnnlGdevWlaenpypXrqyHHnrIrv6+e/duWSwWTZw4sdCY69atk8Vi0RdffGFrO3jwoPr166fAwEBZrVY1aNBAs2bNKvTa999/Xw0aNFD58uVVqVIlNWnSRHPnzr3q+7vY+vXrFRcXp/79+xcKRiTJarXq3XfftWtbuXKlWrRoIS8vL1WsWFGdOnXS1q1b7fqMGzdOFotFO3fuVJ8+fVSxYkX5+vqqb9++OnPmjKTz2QGLxaJVq1Zpy5YtttLG6tWrbaWOi3/rL3jNhSWKlJQU9e3bV1WrVpXValVwcLA6depk9//nUmtIjhw5ov79+yswMFDlypVTo0aNNHv27Ete791339VHH32kWrVqyWq16o477tCvv/5axO+y1KlTJ1mtVs2fP9+ufe7cuerevfslg4OivIdq1qypLVu2aM2aNbbvX8F9FpRa16xZo2eeeUYBAQGqWrWq3bmC79HKlSvl5uamMWPGFJqfxWK55HsXKCnIkFyHQkND1atXL3388ccaNWrUFbMk06dPV4MGDfTAAw+oTJkyWrx4sZ555hnl5+crOjr6kq8JDAxUq1atNG/ePI0dO9bu3FdffSV3d3c99NBDkqQzZ86oVatWOnjwoJ5++mlVr15d69at0+jRo3X48GFNmjTJ4f3k5eUpLS1NkpSbm6utW7dq7Nixql27tpo3b27r9+uvv2rdunXq0aOHqlatqr1792r69Olq3bq1/vzzT5UvX1433nijmjdvrjlz5hRaAzFnzhxVqFBBnTp1knQ+8GrWrJksFosGDRokf39/LV26VP3791dmZqaGDBkiSfr444/17LPPqlu3bnruueeUlZWljRs3av369Xr00Ucd3l9RLFq0SJLUs2fPIvX//vvvdd999+nGG2/UuHHjdPbsWb3//vtq3ry5NmzYoJo1a9r17969u0JDQxUTE6MNGzZo5syZCggI0Pjx4+Xv76/PPvtMb775pk6dOmUrn9WvX79QgHMlXbt21ZYtWzR48GDVrFlTR44cUXx8vJKTkwvNp8DZs2fVunVr7dy5U4MGDVJoaKjmz5+vPn36KD09Xc8995xd/7lz5+rkyZN6+umnZbFYNGHCBHXp0kW7d+9W2bJlHc6xfPny6tSpk7744gsNHDhQkvTHH39oy5YtmjlzpjZu3FjoNUV5D02aNEmDBw+Wt7e3LZMVGBhoN84zzzwjf39/jRkzRqdPn77k/O655x4988wziomJUefOnXX77bfr8OHDGjx4sCIiIjRgwACH9wi4jIHrxqeffmpIMn799Vdj165dRpkyZYxnn33Wdr5Vq1ZGgwYN7F5z5syZQuNERkYaN954o11bq1atjFatWtm+/vDDDw1JxqZNm+z6hYWFGffcc4/t69dff93w8vIyduzYYddv1KhRhru7u5GcnHzFe2rVqpUhqdBRv359Y/fu3Q7vJSEhwZBk/Pvf/y40961bt9racnJyjCpVqhi9e/e2tfXv398IDg420tLS7Mbs0aOH4evra7tep06dCn1fne3BBx80JBknTpwoUv9bb73VCAgIMI4dO2Zr++OPPww3NzejV69etraxY8cakox+/foVul7lypXt2i7187Nq1SpDkrFq1Sq79j179hiSjE8//dQwDMM4ceKEIcl45513rjjvi3/OJk2aZEgyPv/8c1tbTk6OER4ebnh7exuZmZl216tcubJx/PhxW9//+7//MyQZixcvvuJ1C+5j/vz5xpIlSwyLxWL72Rw+fLjt/fB33kMNGjSwu7cCBe/bu+++2zh37twlz+3Zs8fWdvr0aaN27dpGgwYNjKysLCMqKsrw8fEx9u3bd8V7BFyNks116sYbb1TPnj310Ucf6fDhw5ft5+npaftzRkaG0tLS1KpVK+3evVsZGRmXfV2XLl1UpkwZffXVV7a2zZs3688//9TDDz9sa5s/f75atGihSpUqKS0tzXZEREQoLy9Pa9eudXgvNWvWVHx8vOLj47V06VJNmjRJGRkZuu+++3T06NFL3ktubq6OHTum2rVrq2LFitqwYYPtXPfu3VWuXDnNmTPH1rZ8+XKlpaXZ1qoYhqGvv/5aHTt2lGEYdnOPjIxURkaGbcyKFSvqwIEDxSoNFFdmZqYkqUKFCg77Hj58WElJSerTp4/8/Pxs7bfccovuvffeQgs2JRX6zbpFixY6duyY7bp/l6enpzw8PLR69epircH49ttvFRQUpEceecTWVrZsWT377LM6deqU1qxZY9f/4YcfVqVKlWxft2jRQtL5Ul1RtWvXTn5+fvryyy9lGIa+/PJLu+tf7GrfQxd78skni7RepHz58oqNjdXWrVvVsmVLxcXFaeLEiapevXqRrwW4AgHJdezll1/WuXPnrriW5KefflJERIRtnYG/v79efPFFSbriX6ZVqlRR27ZtNW/ePFvbV199pTJlyqhLly62tr/++kvLli2Tv7+/3RERESHp/PoAR7y8vBQREaGIiAi1b99ezz33nBYtWqTt27fb3dvZs2c1ZswY21qVKlWqyN/fX+np6Xb3UrFiRXXs2NFujcecOXN0ww036J577pEkHT16VOnp6froo48Kzb1gJ1PB3EeOHClvb2/deeedqlOnjqKjo/XTTz85vK+UlBS74+zZs5ft6+PjI0k6efKkw3H37dsnSapbt26hc/Xr11daWlqhksDF/5gV/KPurAWcVqtV48eP19KlSxUYGKiWLVtqwoQJSklJueLr9u3bpzp16sjNzf6vsvr169vOX8gZ91G2bFk99NBDmjt3rtauXav9+/dfsfR2te+hi4WGhha5b/PmzTVw4ED98ssvioyMVL9+/Yr8WsBVCEiuYzfeeKMef/zxy2ZJdu3apbZt2yotLU3/+te/FBcXp/j4eNvaCkeLTnv06KEdO3YoKSlJkjRv3jy1bdvWbndCfn6+7r33XluG4+LjUgs0i6Jx48by9fW1y7AMHjxYb775prp376558+bpu+++U3x8vCpXrlzoXnr16qXdu3dr3bp1OnnypBYtWqRHHnnE9g9fQf/HH3/8snMvWL9Sv35929bbu+++W19//bXuvvvuQutrLhYcHGx3XJhtuli9evUkSZs2bSr+N6sILvebueHgqQGX24qal5dXqG3IkCHasWOHYmJiVK5cOb3yyiuqX7++fv/99+JP+DKu9j4u9uijjyopKUnjxo1To0aNFBYWdsl+f/c9dKELMy2OZGdn2xYS79q1y7YAGSjJWNR6nXv55Zf1+eefa/z48YXOLV68WNnZ2Vq0aJHdb5arVq0q0tidO3fW008/bfuHdMeOHRo9erRdn1q1aunUqVO2jIgz5eXl6dSpU7avFyxYoN69e+u9996ztWVlZSk9Pb3Qa9u3by9/f3/NmTNHTZs21ZkzZ+wWjPr7+6tChQrKy8sr0ty9vLz08MMP6+GHH1ZOTo66dOmiN998U6NHj1a5cuUu+Zr4+Hi7rxs0aHDZ8Tt27KiYmBh9/vnntjLE5dSoUUPS+R1P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LtknODhY69evV8uWLSVJ586dU2Jiom6//fZL9m/YsKHy8/O1Zs0aRUREFDpfkKHJy8uztYWFhclqtSo5OfmymZX69etr0aJFdm0///yz45sEALGoFSjRHnvsMVWpUkWdOnXSDz/8oD179mj16tV69tlndeDAAUnSc889p7ffflsLFy7Utm3b9Mwzz1zxGSI1a9ZU79691a9fPy1cuNA25rx58yRJNWrUkMVi0ZIlS3T06FGdOnVKFSpU0AsvvKChQ4dq9uzZ2rVrlzZs2KD3339fs2fPliQNGDBAf/31l4YPH67t27dr7ty5io2NNftbBKCUICABSrDy5ctr7dq1ql69urp06aL69eurf//+ysrKsmVMnn/+efXs2VO9e/dWeHi4KlSooAcffPCK406fPl3dunXTM888o3r16unJJ5/U6dOnJUk33HCDXn31VY0aNUqBgYEaNGiQJOn111/XK6+8opiYGNWvX1/t27dXXFycQkNDJUnVq1fX119/rYULF6pRo0aaMWOG3nrrLRO/OwBKE4txuZVvAAAA1wgZEgAA4HIEJAAAwOUISAAAgMsRkAAAAJcjIAEAAC5HQAIAAFyOgAQAALgcAQkAAHA5AhIAAOByBCQAAMDlCEgAAIDLEZAAAACX+39kTah1LqHFEgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "===== Logistic Regression =====\n",
            "Accuracy:  0.9868\n",
            "Precision: 0.9840\n",
            "Recall:    0.9884\n",
            "F1 Score:  0.9862\n",
            "\n",
            "Classification Report:\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.99      0.99      0.99      5871\n",
            "           1       0.98      0.99      0.99      5354\n",
            "\n",
            "    accuracy                           0.99     11225\n",
            "   macro avg       0.99      0.99      0.99     11225\n",
            "weighted avg       0.99      0.99      0.99     11225\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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e2zdunX4999/DfaV5GdaSnh4OBITE7F8+XJ8++23qFKlivgwQ6LiYoWkFPjxxx+xY8eOQvtHjBiBadOmic9K+PTTT2FmZoZFixZBp9MhKipKbDt27Fj89NNPeO+99zBs2DBx2W/lypVx586dZ/415O/vD1dXVzRt2hQuLi44d+4c5s2bh8DAQHHSoo+PDwDg888/R/fu3WFubo4OHToU+dfgmDFjsH79enTr1g39+/eHj48P7ty5gy1btmDhwoWoX79+id+jtm3bok6dOvj2228REhKCli1bYtCgQYiMjERCQgL8/f1hbm6OixcvYt26dZgzZw66du0KFxcXjBgxAjNnzsT777+PNm3a4OTJk9i+fTsqVKhQrL8SNRoNFixYgN69e6Nhw4bo3r07nJyckJiYiJiYGDRt2hTz5s3D33//jdatW+ODDz6At7c3zMzMsGnTJqSkpIh/0S5fvhzz589H586dUa1aNdy7dw8//PADNBrNM/8xK+73wctUrVo1rFq1Ch9++CFq165t8KTWw4cPY926deJzXurXr4/g4GAsXrwY6enpaNmyJf78808sX74cnTp1gq+vr9Hi6t69O8LDw9G5c2cMHz5cXKL9xhtvGEzqnDJlCg4cOIDAwEB4enri1q1bmD9/Pjw8PNCsWbOn9j9jxgy0bdsWWq0WAwYMEJf92tvby/r5QyYmJvjiiy8k27Vv3x5TpkxBv3798J///AenT5/GypUrDSZEA4++fg4ODli4cCHs7OxgY2ODxo0bF5pDJWXPnj2YP38+Jk6cKC5DXrZsGVq1aoXx48cr9v1JryDlFvhQwRLNp23Xr18XBEEQjh8/LgQEBAi2traCtbW14OvrKxw+fLhQfydOnBCaN28uqNVqwcPDQ4iMjBTmzp0rABCSk5PFdk8u21u0aJHQokULoXz58oJarRaqVasmjBkzRsjIyDDof+rUqULFihUFExMTgyXATy77FYRHS46HDh0qVKxYUbCwsBA8PDyE4OBgIS0t7ZnviaenpxAYGFjksejo6ELLFBcvXiz4+PgIVlZWgp2dnVC3bl1h7NixQlJSktjm4cOHwvjx4wVXV1fByspKePfdd4Vz584J5cuXFwYPHlzo6/G0Jbl79+4VAgICBHt7e8HS0lKoVq2a0LdvX+HYsWOCIAhCWlqaEBISItSqVUuwsbER7O3thcaN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uAAAFi5ciPDwcKSmpsLCwgLh4eGIiYnBmTNnxGt2794d6enp2LFjR7HjZIWEiIhIbnq9UTadTofMzEyDTafTPfPSISEhCAwMhJ+fn8H++Ph45OXlGeyvVasWKleujLi4OABAXFwc6tatKyYjABAQEIDMzEycPXtWbPNk3wEBAWIfxcWEhIiI6BURGRkJe3t7gy0yMvKp7X/++WccP368yDbJycmwsLCAg4ODwX4XFxckJyeLbR5PRgqOFxx7VpvMzEw8ePCg2PfGVTZERERyM9Iqm4iICISFhRnsU6vVRba9fv06RowYgdjYWFhaWhrl+nJihYSIiEhuesEom1qthkajMdielpDEx8fj1q1baNiwIczMzGBmZob9+/dj7ty5MDMzg4uLC3Jzc5Genm5wXkpKClxdXQEArq6uhVbdFLyWaqPRaGBlZVXst4gJCRERURnUunVrnD59GgkJCeLWqFEj9OrVS/x/c3Nz7N69WzznwoULSExMhFarBQBotVqcPn0at27dEtvExsZCo9HA29tbbPN4HwVtCvooLg7ZEBERyUyJBa12dnaoU6eOwT4bGxuUL19e3D9gwACEhYXB0dERGo0Gw4YNg1arRZMmTQAA/v7+8Pb2Ru/evREVFYXk5GR88cUXCAkJESszgwcPxrx58zB27Fj0798fe/bswdq1axETE1OieJmQEBERya2UPql11qxZMDExQZcuXaDT6RAQEID58+eLx01NTbFt2zYMGTIEWq0WNjY2CA4OxpQpU8Q2Xl5eiImJQWhoKObMmQMPDw8sWbIEAQEBJYqFzyEheo3wOSREhb2M55BkDnjPKP1olsYapZ/SiHNIiIiISHEcsiEiIpLZi34OzeuACQkREZHcmJBI4pANERERKY4VEiIiIrnplQ6g9GNCQkREJDPOIZHGIRsiIiJSHCskREREcmOFRBITEiIiIrlxDokkDtkQERGR4lghISIikhkntUpjQkJERCQ3DtlIYkJCREQkM1ZIpHEOCRERESmOFRIiIiK5cchGEhMSIiIimQlMSCRxyIaIiIgUxwoJERGR3FghkcSEhIiISGYcspHGIRsiIiJSHCskREREcmOFRBITEiIiIplxyEYaExIiIiKZMSGRxjkkREREpDhWSIiIiGTGCok0JiRERERyE1RKR1DqcciGiIiIFMcKCRERkcw4ZCONCQkR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VTavmhLPIdm7d68ccRAREZVZeqUDeAWUeJVNly5d8PXXXxfaHxUVhW7duhklKCIiInq9lDghOXDgANq1a1dof9u2bXHgwAGjBEVERFSWCFAZZSvLSjxkk5WVVeTyXnNzc2RmZholKCIiorJEz3W/kkpcIalbty7WrFlTaP/PP/8sftgOERER/Y8eKqNsZVmJKyTjx49HUFAQLl++jHfffRcAsHv3bqxatQrr1683eoBERERU9pU4IenQoQM2b96M6dOnY/369bCyskL9+vWxZ88eODo6yhEjERHRK62sz/8whhInJAAQGBiIwMBAAEBmZiZWr16N0aNHIz4+Hvn5+UYNkIiI6FXHZb/SSjyHpMCBAwcQHBwMd3d3zJw5E++++y6OHDlizNiIiIjoNVGiCklycjKio6OxdOlSZGZm4oMPPoBOp8PmzZs5oZWIiOgpOGQjrdgVkg4dOqBmzZo4deoUZs+ejaSkJHz33XdyxkZERFQm6I20lWXFrpBs374dw4cPx5AhQ1CjRg05YyIiIqLXTLErJAcPHsS9e/fg4+ODxo0bY968eUhLS5MzNiIiojKBFRJpxU5ImjRpgh9++AE3b97EoEGD8PPPP8Pd3R16vR6xsbG4d++enHESERG9svjoeGklXmVjY2OD/v374+DBgzh9+jRGjRqFr776Cs7Oznj//ffliJGIiIjKuOde9gsANWvWRFRUFG7cuIHVq1cbKyYiIqIyRa8yzlaWPdeD0Z5kamqKTp06oVOnTsbojoiIqEwp659DYwxGSUiIiIjo6fhhv9JeaMiGiIiIyBhYISEiIpJZWV+yawxMSIiIiGSmV3EOiRQO2RAREZHiWCEhIiKSGSe1SmNCQkREJDPOIZHGIRsiIiJSHCskREREMivrT1k1BiYkREREMuOTWqVxyIaIiIgUxwoJERGRzLjKRhoTEiIiIplxDok0JiREREQy47JfaZxDQkRERIpjhYSIiEhmnEMijQkJERGRzDiHRBqHbIiIiEhxrJAQERHJjJNapTEhISIikhkTEmkcsiEiIiLFsUJCREQkM4GTWiUxISEiIpIZh2ykcciGiIiIFMcKCRERkcxYIZHGhISIiEhmfFKrNCYkREREMuOTWqVxDgkREREpjhUSIiIimXEOiTQmJERERDJjQiKNQzZERESkOCYkREREMhOMtJVEZGQk3n77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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Cross-Validation"
      ],
      "metadata": {
        "id": "TJaQbm5Cz_tr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "cv_scores = cross_val_score(best_lr, X, y, cv=5, scoring='f1')\n",
        "\n",
        "print(\"Cross-validation F1 scores:\", cv_scores)\n",
        "print(\"Average CV F1 score:\", cv_scores.mean())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "azJuWlpe0BRg",
        "outputId": "a3cbc3e9-c666-41ce-c52a-737cf14d2eeb"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cross-validation F1 scores: [0.98598786 0.98775796 0.9811803  0.98796588 0.98520676]\n",
            "Average CV F1 score: 0.9856197501404296\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# PROBABILITY DISTRIBUTION"
      ],
      "metadata": {
        "id": "huwoHTOx19wk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure()\n",
        "sns.histplot(best_lr.predict_proba(X_test)[:,1], bins=50)\n",
        "plt.title(\"Prediction Probability Distribution\")\n",
        "plt.xlabel(\"Probability of Real News\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "TBbDuQek1_k7",
        "outputId": "112723c7-1684-4582-f848-7848d42fe128"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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r7iD4ZXjbIzR/v377ca5YsWKOnnyZK7HXnkUpzC9hYSE6MiRI7lOIV7tPQbrIhAB+ejbt6+ysrL06quv5lp36dIl23/eERERcnNz05w5c+z+kpw1a9Y1n+POO+9UaGioZs2aleuXweXbyvlMpLx+YVzOxcVFnTt31hdffGF3OiItLU2LFy9W69at5evre82+HKFbt25KTU21u5rt0qVLmjNnjnx8fNSuXTu7+uXLl9tdkv7vf/9b3333nbp27Srpf3/dX/nXfH6vQ87HGeSYM2eOJNm2WVQ586eu9nqFhYUpLCxMCxYs0GeffaaHH374uj8raMOGDXr11VcVGhpq+9iIvBw/fjzXspwPOMzIyJBU8PdbQeVcIZjj008/VUpKit0416pVS1u3brW7MnPFihW5Ls8vTG/dunVTVlaW3n77bbvlM2fOlJOT03W/zrh5cMoMyEe7du30xBNPaMqUKUpOTlbnzp3l5uamn376SUuXLtXs2bPVp08fValSRc8995ymTJmiHj16qFu3btq+fbu+/vrra55KcnZ21ty5c9WzZ081adJEgwcPVmBgoPbu3atdu3ZpzZo1kqSmTZtKkp5++mlFRkbKxcVFDz/8cJ7bfO2112yfM/PUU0/J1dVV7777rjIyMjRt2rTiHaTrMGzYML377rsaNGiQkpKSVKNGDX366afavHmzZs2alWveR+3atdW6dWsNHz5cGRkZmjVrlipXrqznn39ekuTr66u2bdtq2rRpunjxoqpVq6a1a9faPg8pLwcOHNC9996rLl26KDExUR9++KEeffRR29G/ovLy8lL9+vX18ccfq06dOqpUqZIaNmxoN1dswIABeu655ySp0KfLvv76a+3du1eXLl1SWlqaNmzYoHXr1ikkJERffvllvh/WOWnSJH377bfq3r27QkJCdPToUb3zzjuqXr26WrduLemvcFKhQgXNmzdP5cuXl7e3t1q0aKHQ0NAijIZUqVIltW7dWoMHD1ZaWppmzZql2rVr2300wNChQ/Xpp5+qS5cu6tu3r37++Wd9+OGHueakFaa3nj17qkOHDnr55Zd18OBBNW7cWGvXrtUXX3yhkSNH5jnfDRZVile4ASWuIJcoG/PXpbre3t5XXf/ee++Zpk2bGi8vL1O+fHnTqFEj8/zzz5sjR47YarKysszEiRNNYGCg8fLyMu3btzc7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tqpXr64FCxYoMjKyNHYJAADcgJyMMaa0m7jRpaeny8/PT6dOnSqR+USVqvir8+Tl+dasfbmXjh9Ly7cGAAD8T2F+f5f65xABAACUNgIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwvBv6gxkBAEDZ1yCsiVJSUvKtCQwM1K4fkx3TUB4IRAAAoESlpKQU6AOISxOnzAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOURiAAAgOWVaiCaO3euwsLC5OvrK19fX4WHh+vrr7+2rb9w4YKioqJUuXJl+fj4qHfv3kpLS7PbxqFDh9S9e3eVK1dOVatW1ZgxY3Tp0iW7moSEBN15553y8PBQ7dq1FRcX54jdAwAAZUSpBqLq1atr6tSpSkpK0vfff6+OHTvqvvvu065duyRJo0aN0ldffaWlS5dq48aNOnLkiB544AHb47OystS9e3dlZmZqy5Yt+uCDDxQXF6fx48fbag4cOKDu3burQ4cOSk5O1siRIzV06FCtWbPG4fsLAABuTE7GGFPaTVyuUqVKeuONN9SnTx9VqVJFixcvVp8+fSRJe/fuVb169ZSYmKi7775bX3/9tXr06KEjR47I399fkjRv3jy98MILOnbsmNzd3fXCCy9o5cqV2rlzp+05Hn74YZ08eVKrV68uUE/p6eny8/PTqVOn5OvrW/z7XMVfnScvz7dm7cu9dPxYWr41AADciErr91xhfn/fMHOIsrKytGTJEp09e1bh4eFKSkrSxYsXFRERYau5/fbbdeuttyoxMVGSlJiYqEaNGtnCkCRFRkYqPT3ddpQpMTHRbhs5NTnbAAAAcC3tBnbs2KHw8HBduHBBPj4+WrZsmerXr6/k5GS5u7urQoUKdvX+/v5KTU2VJKWmptqFoZz1Oevyq0lPT9f58+fl5eWVq6eMjAxlZGTY7qenp1/3fgIAgBtXqR8hqlu3rpKTk/Xdd99p+PDhGjhwoHbv3l2qPU2ZMkV+fn62W3BwcKn2AwAASlapByJ3d3fVrl1bTZs21ZQpU9S4cWPNnj1bAQEByszM1MmTJ+3q09LSFBAQIEkKCAjIddVZzv1r1fj6+uZ5dEiSxo4dq1OnTtluhw8fLo5dBQAAN6hSD0RXys7OVkZGhpo2bSo3NzfFx8fb1u3bt0+HDh1SeHi4JCk8PFw7duzQ0aNHbTXr1q2Tr6+v6tevb6u5fBs5NTnbyIuHh4ftowBybgAA4OZVqnOIxo4dq65du+rWW2/V6dOntXjxYiUkJGjNmjXy8/PTkCFDNHr0aFWqVEm+vr4aMWKEwsPDdffdd0uSOnfurPr166t///6aNm2aUlNTNW7cOEVFRcnDw0OS9OSTT+rtt9/W888/r8cee0wbNmzQJ598opUrV5bmrgMAgBtIqQaio0ePasCAAUpJSZGfn5/CwsK0Zs0a3XPPPZKkmTNnytnZWb1791ZGRoYiIyP1zjvv2B7v4uKiFStWaPjw4QoPD5e3t7cGDhyoSZMm2WpCQ0O1cuVKjRo1SrNnz1b16tW1YMECRUZGOnx/AQDAjalUA9E//vGPfNd7enoqNjZWsbGxV60JCQnRqlWr8t1O+/bttX379iL1CAAAbn433BwiAAAARyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyyMQAQAAyytSIKpZs6b+/PPPXMtPnjypmjVrXndTAAAAjlSkQHTw4EFlZWXlWp6RkaHff//9upsCAABwJNfCFH/55Ze2f69Zs0Z+fn62+1lZWYqPj1eNGjWKrTkAAABHKFQg6tWrlyTJyclJAwcOtFvn5uamGjVqaPr06cXWHAAAgCMUKhBlZ2dLkkJDQ7Vt2zbdcsstJdIUAACAIxUqEOU4cOBAcfcBAABQaooUiCQpPj5e8fHxOnr0qO3IUY7333//uhsDAABwlCIFookTJ2rSpElq1qyZAgMD5eTkVNx9AQAAOEyRAtG8efMUFxen/v37F3c/AAAADlekzyHKzMxUy5Yti7sXAACAUlGkQDR06FAtXry4uHsBAAAoFUU6ZXbhwgW99957Wr9+vcLCwuTm5ma3fsaMGcXSHAAAgCMUKRD9+OOPatKkiSRp586dduuYYA0AAMqaIgWib775prj7AAAAKDVFmkMEAABwMynSEaIOHTrke2psw4YNRW4IAADA0YoUiHLmD+W4ePGikpOTtXPnzlxf+goAAHCjK1IgmjlzZp7LY2JidObMmetqCAAAwNGKdQ7R3/72N77HDAAAlDnFGogSExPl6elZnJsEAAAocUU6ZfbAAw/Y3TfGKCUlRd9//71eeeWVYmkMAADAUYoUiPz8/OzuOzs7q27dupo0aZI6d+5cLI0BAAA4SpEC0cKFC4u7DwAAgFJTpECUIykpSXv27JEkNWjQQHfccUexNAUAAOBIRQpER48e1cMPP6yEhARVqFBBknTy5El16NBBS5YsUZUqVYqzRwAAgBJVpKvMRowYodOnT2vXrl06fvy4jh8/rp07dyo9PV1PP/10cfcIAABQoop0hGj16tVav3696tWrZ1tWv359xcbGMqkaAACUOUU6QpSdnS03N7dcy93c3JSdnX3dTQEAADhSkQJRx44d9cwzz+jIkSO2Zb///rtGjRqlTp06FVtzAAAAjlCkQPT2228rPT1dNWrUUK1atVSrVi2FhoYqPT1dc+bMKe4eAQAASlSR5hAFBwfrhx9+0Pr167V3715JUr169RQREVGszQEAADhCoY4QbdiwQfXr11d6erqcnJx0zz33aMSIERoxYoTuuusuNWjQQP/6179KqlcAAIASUahANGvWLD3++OPy9fXNtc7Pz09PPPGEZsyYUWzNAQAAOEKhAtF//vMfdenS5arrO3furKSkpOtuCgAAwJEKFYjS0tLyvNw+h6urq44dO3bdTQEAADhSoQJRtWrVtHPnzquu//HHHxUYGHjdTQEAADhSoQJRt27d9Morr+jChQu5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Urdu3RzSsxWU1u9BS31StaP98ccfysrKyvX1IP7+/tq7d2+ej0lNTc2zPjU1tcT6LOuKMs5XeuGFFxQUFJTrhxD/U5Rx3rRpk/7xj38oOTnZAR3eHIoyzr/88os2bNigfv36adWqVdq/f7+eeuopXbx4URMmTHBE22VSUcb60Ucf1R9//KHWrVvLGKNLly7pySef1EsvveSIli3har8H09PTdf78eXl5eZXI83KECJY3depULVmyRMuWLZOnp2dpt3PTOH36tPr376/58+frlltuKe12bmrZ2dmqWrWq3nvvPTVt2lQPPfSQXn75Zc2bN6+0W7vpJCQk6PXXX9c777yjH374QZ9//rlWrlypV199tbRbw3XiCFEJuuWWW+Ti4qK0tDS75WlpaQoICMjzMQEBAYWqR9HGOcebb76pqVOnav369QoLCyvJNsu8wo7zzz//rIMHD6pnz562ZdnZ2ZIkV1dX7du3T7Vq1SrZpsugoryfAwMD5ebmJhcXF9uyevXqKTU1VZmZmXJ3dy/Rnsuqooz1K6+8ov79+2vo0KGSpEaNGuns2bMaNmyYXn75ZTk7c5zhel3t96Cvr2+JHR2SOEJUotzd3dW0aVPFx8fblmVnZys+Pl7h4eF5PiY8PNyuXpLWrVt31XoUbZwladq0aXr11Ve1evVqNWvWzBGtlmmFHefbb79dO3bsUHJysu127733qkOHDkpOTlZwcLAj2y8zivJ+btWqlfbv328LnJL03//+V4GBgYShfBRlrM+dO5cr9OQEUcNXgxaLUvs9WKJTtmGWLFliPDw8TFxcnNm9e7cZNmyYqVChgklNTTXGGNO/f3/z4osv2uo3b95sXF1dzZtvvmn27NljJkyYwGX3BVDYcZ46dapxd3c3n376qUlJSbHdTp8+XVq7UCYUdpyvxFVmBVPYcT506JApX768iY6ONvv27TMrVqwwVatWNa+99lpp7UKZUdixnjBhgilfvrz56KOPzC+//GLWrl1ratWqZfr27Vtau3DDO336tNm+fbvZvn27kWRmzJhhtm/fbn799VdjjDEvvvii6d+/v60+57L7MWPGmD179pjY2Fguu79ZzJkzx9x6663G3d3dNG/e3GzdutW2rl27dmbgwIF29Z988ompU6eOcXd3Nw0aNDArV650cMdlU2HGOSQkxEjKdZswYYLjGy9jCvt+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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Model Comparison Visualization"
      ],
      "metadata": {
        "id": "9pJivcH10F1C"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "models = ['Naive Bayes', 'Logistic Regression']\n",
        "f1_scores = [nb_results[3], lr_results[3]]\n",
        "\n",
        "plt.figure()\n",
        "plt.bar(models, f1_scores)\n",
        "plt.title(\"Model Comparison (F1 Score)\")\n",
        "plt.ylabel(\"F1 Score\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 452
        },
        "id": "bcukovcG0HGJ",
        "outputId": "5f3c11a9-6f7d-4546-8ba5-dc7ff6f6d7ae"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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T0emp6dr6dKlatasmWrUqJHldqS/f261atVymJTv7e2tnj176tChQ/r9998d1uvcubPDzzc2NlZ79+5Vhw4ddOrUKfu//+TkZDVs2FDff/99plGrQoUK6eLFizk6BYj7GxOKcd8qVqyYIiIiNH/+fF24cEFpaWnXPaVz+PBhlShRIlN4qFChgn15xn9dXV1VpkwZh37ly5d3eH7ixAmdPXtWM2bM0IwZM7Lc5/Hjx3N1PD4+PlmeDspKRr3X1uXu7q7SpUvbl2d44IEHMl2h4uvrq6CgoExt0t+nsa5Vrlw5h+dlypSRq6urw2mfjRs3Kjo6Wps3b870CywxMdG+fenvcJMTI0aMUIsWLfTQQw+pUqVKaty4sV588UVVqVJF0vVfC+nvn++qVauUnJysAgUK2NtLlSrl0C8jsJw5c0Y+Pj7Z1mOumaN1tcqVK9/0fJycsNlsGjx4sAYPHqxjx45pw4YNmjJlir744gvlz59fn376qU6cOKGkpCRVqlQp220dPnzYIdhmuPo9cfU2rv15ZYTNzp07X3cfiYmJDmEw47XjaincCOEG97UOHTqoR48eio+PV5MmTeTn53dH9pvxF+kLL7xw3Q/3jF++ORUaGqrY2FilpqbK3d39lmu8mpubW67as/sFnuHaX1D79+9Xw4YNFRoaqokTJyooKEju7u5asWKFJk2alOmv+JxOrK1fv77279+vr776SqtXr9asWbM0adIkTZ8+XS+99FKOtnGtmz3uIkWKZBn8nCEwMFDt2rXTc889p4cfflhffPHFbb1R5LU/r4yf5zvvvHPd+VLe3t4Oz8+cOSMvL6+bnlSN+wfhBve1Vq1a6f/+7//0008/acGCBdftFxwcrLVr1+rcuXMOozcZNz7LmKwZHBys9PR07d+/32EkYM+ePQ7by7iSKi0tLc/+Um/WrJk2b96sxYsXq3379tn2zah3z549Kl26tL09NTVVBw8evC2jB3v37nX4633fvn1KT0+338flm2++UUpKir7++muHkZF169bd8r4LFy6srl27qmvXrjp//rzq16+vYcOG6aWXXnJ4La61e/duFS1a1GHU5laEhoZq8eLFebKtvJI/f35VqVJFe/fu1cmTJ1W8eHH5+Pjo119/zXa94ODg675mGcuzkzG66ePjk+N/bwcPHrSPDAHZYc4N7mve3t768MMPNWzY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bM4uAwAA3CH31MhNQECAEhISHNoSEhLk4+MjT09PJ1UFAADuJvdUuAkPD1dMTIxD25o1axQeHu6kigAAwN3Gqaelzp8/r3379tmfHzx4ULGxsSpcuLBKlSqlQYMG6ejRo/rXv/4lSXr55Zc1depU/fOf/1S3bt303Xff6YsvvtDy5cuddQgA7kMhA/nMAbJzaNzTTt2/U0dutm3bpkceeUSPPPKIJCkqKkqPPPKIhg4dKkk6duyYjhw5Yu//4IMPavny5VqzZo2qVq2qCRMmaNasWde9DBwAANx/XIwxxtlF3ElJSUny9fVVYmKifHx8nF0OgHsQIzdA9m7HyE1ufn/fU3NuAAAAboRwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALCWfswuwmpCBy51dAnDXOjTuaWeXAOA+wMgNAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFKeHm2nTpikkJEQeHh6qXbu2tmzZkm3/yZMnq3z58vL09FRQUJBee+01Xbp06Q5VCwAA7nZODTcLFixQVFSUoqOjtWPHDlWtWlWRkZE6fvx4lv3nz5+vgQMHKjo6Wrt27dLHH3+sBQsW6M0337zDlQMAgLuVU8PNxIkT1aNHD3Xt2lUVK1bU9OnT5eXlpdmzZ2fZf9OmTapbt646dOigkJAQNWrUSO3bt7/haA8AALh/OC3cpKamavv27YqIiPhfMa6uioiI0ObNm7Ncp06dOtq+fbs9zBw4cEArVqxQ06ZNr7uflJQUJSUlOTwAAIB15XPWjk+ePKm0tDT5+/s7tPv7+2v37t1Z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MQAAALcqx+GmZcuWcnFxkTHmun1cXFzypCgAAICblePTUoGBgfryyy+Vnp6e5WPHjh23s04AAIAcyXG4CQsLs38reFZuNKoDAABwJ+T4tNQbb7yh5OTk6y4vW7as1q1blydFAQAA3Kwch5t69eplu7xAgQJq0KDBLRcEAABwK3J8WurAgQOcdgIAAHe9HIebcuXK6cSJE/bnbdu2VUJCwm0pCgAA4GblONxcO2qzYsWKbOfgAAAAOAPfLQUAACwlx+HGxcUl0036uGkfAAC42+T4ailjjLp06SKbzSZJunTpkl5++WUVKFDAod+XX36ZtxUCAADkQo7DTefOnR2ev/DCC3leDAAAwK3KcbiZM2fO7awDAAAgTzChGAAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWMpdEW6mTZumkJAQeXh4qHbt2tqyZUuO1vv888/l4uKili1b3t4CAQDAPcPp4WbBggWKiopSdHS0duzYoapVqyoyMlLHjx/Pdr1Dhw7pH//4h+rVq3eHKgUAAPcCp4ebiRMnqkePHuratasqVqyo6dOny8vLS7Nnz77uOmlpaerYsaOGDx+u0qVL38FqAQDA3c6p4SY1NVXbt29XRESEvc3V1VURERHavHnzddcbMWKEihcvru7du99wHykpKUpKSnJ4AAAA63JquDl58qTS0tLk7+/v0O7v76/4+Pgs1/nxxx/18ccfa+bMmTnax9ixY+Xr62t/BAUF3XLdAADg7uX001K5ce7cOb344ouaOXOmihYtmqN1Bg0apMTERPsjLi7uNlcJAACcKcffCn47FC1aVG5ubkpISHBoT0hIUEBAQKb++/fv16FDh9SsWTN7W3p6uiQpX7582rNnj8qUKeOwjs1mk81muw3VAwCAu5FTR27c3d0VFhammJgYe1t6erpiYmIUHh6eqX9oaKj++9//KjY21v5o3ry5nnjiCcXGxnLKCQAAOHfkRpKioqLUuXNn1ahRQ7Vq1dLkyZOVnJysrl27SpI6deqkkiVLauzYsfLw8FClSpUc1vfz85OkTO0AAOD+5PRw07ZtW504cUJDhw5VfHy8qlWrppUrV9onGR85ckSurvfU1CAAAOBETg83ktSnTx/16dMny2Xr16/Pdt25c+fmfUEAAOCexZAIAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwFMINAACwlLsi3EybNk0hISHy8PBQ7dq1tWXLluv2nTlzpurVq6dChQqpUKFCioiIyLY/AAC4vzg93CxYsEBRUVGKjo7Wjh07VLVqVUVGRur48eNZ9l+/fr3at2+vdevWafPmzQoKClKjRo109OjRO1w5AAC4Gzk93EycOFE9evRQ165dVbFiRU2fPl1eXl6aPXt2lv3nzZunXr16qVq1agoNDdWsWbOUnp6umJiYO1w5AAC4Gzk13KSmpmr79u2KiIiwt7m6uioiIkKbN2/O0TYuXLigy5cvq3DhwlkuT0lJUVJSksMDAABYl1PDzcmTJ5WWliZ/f3+Hdn9/f8XHx+doGwMGDFCJEiUcAtLVxo4dK19fX/sjKCjolusGAAB3L6eflroV48aN0+eff64lS5bIw8Mjyz6DBg1SYmKi/REXF3eHqwQAAHdSPmfuvGjRonJzc1NCQoJDe0JCggICArJd991339W4ceO0du1aValS5br9bDabbDZbntQLAADufk4duXF3d1dYWJjDZOCMycHh4eHXXe/tt9/WyJEjtXLlStWoUeNOlAoAAO4RTh25kaSoqCh17txZNWrUUK1atTR58mQlJyera9eukqROnTqpZMmSGjt2rCRp/PjxGjp0qObPn6+QkBD73Bxvb295e3s77TgAAMDdwenhpm3btjpx4oSGDh2q+Ph4VatWTStXrrRPMj5y5IhcXf83wPThhx8qNTVVrVu3dthOdHS0hg0bdidLBwAA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/38SoiRTsGBBU7NmTbNo0SKHfm+88YYpUqSI8fb2Nm3btjWTJk1ymAiZ4eOPP3aY5He1P/74w7Rq1cr4+fkZT09PExoaavr372/S09PN77//biIjI02xYsWMzWYzDz30kHn//fevWzdwI9f+2854dO/e3RhjzPr1603NmjWNu7u7CQgIMAMGDDCXL1+2r5+UlGQ6dOhgvLy8TEBAgJk4caKpVauWGThwoL3P1ZNelyxZYmrXrm18fHxMgQIFzKOPPmrWrl1r77t582ZTpUoVY7PZ7J8F104oNsaYxYsX29/DRYsWNc8+++x1jzGr9TP2Jcn8/PPPxhhjVq5caerUqWM8PT2Nj4+PqVWrlpkxY4a9/65du0zdunWNu7u7CQ0NNd98842RZFauXGmM+d+E4l9++SXTvrL7zJkxY4apVq2aKVCggPHx8TENGzY0O3bsyNHrde0E7MOHD5vmzZubAgUKmIIFC5rnn3/exMfH25dHR0ebqlWrOtQ2adIkExwcfN3X737kYgzXoAK5NXLkSC1cuJA7HMNykpOTVbJkSU2YMEHdu3d3djm31caNG/XYY49p3759KlOmjLPLQR7itBSQC+fPn9ehQ4c0depUh3uFAPeqX375Rbt371atWrWUmJhov1lmixYtnFxZ3luyZIm8vb1Vrlw57du3T/369VPdunUJNhbE1VJALvTp00dhYWF6/PHHHa6SAu5l7777rqpWrWq/W/YPP/yQ5VyZe925c+fsE3q7dOmimjVr6quvvnJ2WbgNOC0FAAAshZEbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKf8PEbAPOt3brQMAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# DATA LEAKAGE CHECK"
      ],
      "metadata": {
        "id": "TI01ML6b0gwI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Check class distribution\n",
        "print(df['label'].value_counts())\n",
        "\n",
        "# Check average length difference\n",
        "df['length'] = df['content'].apply(lambda x: len(x.split()))\n",
        "\n",
        "print(\"\\nAverage length by class:\")\n",
        "print(df.groupby('label')['length'].mean())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2N-c_mGJ0iCA",
        "outputId": "ea716269-3968-47ca-c5bc-a639daec6e00"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "label\n",
            "0    23481\n",
            "1    21417\n",
            "Name: count, dtype: int64\n",
            "\n",
            "Average length by class:\n",
            "label\n",
            "0    236.022188\n",
            "1    225.229397\n",
            "Name: length, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# ROC Curve"
      ],
      "metadata": {
        "id": "1oD-lwqN0ldI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve, auc\n",
        "\n",
        "y_probs = best_lr.predict_proba(X_test)[:, 1]\n",
        "\n",
        "fpr, tpr, thresholds = roc_curve(y_test, y_probs)\n",
        "roc_auc = auc(fpr, tpr)\n",
        "\n",
        "plt.figure()\n",
        "plt.plot(fpr, tpr, label=f\"AUC = {roc_auc:.4f}\")\n",
        "plt.plot([0, 1], [0, 1], linestyle='--')\n",
        "\n",
        "plt.xlabel(\"False Positive Rate\")\n",
        "plt.ylabel(\"True Positive Rate\")\n",
        "plt.title(\"ROC Curve - Logistic Regression\")\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "IndvByZi0m__",
        "outputId": "a0a8a510-c11a-4323-af3d-0ff0e42e5cd6"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAZ8xJREFUeJzt3XmcjeX/x/HXmX3GLGiMdezZImuEJEzGGslSKluhUEoqLYiKNtK3lCKkzRZSRAghRRgpW7asYwkzttnOuX5/nJ+jaWY0hzNzz/J+Ph7n8Z37Otd9359zz3w7b9d93fdtM8YYRERERPIIL6sLEBEREfEkhRsRERHJUxRuREREJE9RuBEREZE8ReFGRERE8hSFGxEREclTFG5EREQkT1G4ERERkTxF4UZERETyFIUbEcnzbDYbL730kke2deDAAWw2G9OnT/fI9gRWrVqFzWZj1apVVpcieYTCjeR606dPx2azuV4+Pj6ULFmSXr16ceTIkXTXMcbw6aefcvvtt1OwYEGCgoKoUaMGo0eP5sKFCxnua/78+bRu3Zrw8HD8/PwoUaIEXbt25YcffshUrQkJCbz99ts0aNCAsLAwAgICqFSpEoMGDWL37t3X9Plzk169ehEcHGx1GZnyxRdfMGHChCzdx+WgdPnl5eVF4cKFad26NevXr8/SfYvkZTY9W0pyu+nTp9O7d29Gjx5NuXLlSEhI4Oeff2b69OmULVuW33//nYCAAFd/u91O9+7dmT17Nk2aNKFTp04EBQWxZs0avvjiC6pVq8by5cspWrSoax1jDH369GH69OnUrl2bzp07U6xYMY4dO8b8+fPZtGkT69ato1GjRhnWeerUKVq1asWmTZto164dUVFRBAcHs2vXLmbOnElsbCxJSUlZeqys1qtXL+bOncv58+ezdb8JCQn4+Pjg4+OT6XXatWvH77//zoEDB1K1G2NITEzE19cXb2/v66rrwIEDlCtXjvvuu482bdpgt9vZvXs377//PpcuXWLjxo3UqFHjuvaRGzgcDpKSkvDz88PLS//mFg8wIrnctGnTDGA2btyYqv3ZZ581gJk1a1aq9jFjxhjADB06NM22Fi5caLy8vEyrVq1Stb/55psGME888YRxOBxp1psxY4b55Zdfrlpn27ZtjZeXl5k7d26a9xISEsxTTz111fUzKzk52SQmJnpkW57Ws2dPU6BAAavLyJS2bduaMmXKZOk+9u/fbwDz5ptvpmr/7rvvDGAeffTRLN1/es6fP5/t+xTxNIUbyfUyCjfffvutAcyYMWNcbRcvXjSFChUylSpVMsnJyelur3fv3gYw69evd61TuHBhU6VKFZOSknJNNf78888GMH379s1U/6ZNm5qmTZumae/Zs2eqL9x/fjm+/fbbpnz58sbLy8v8/PPPxtvb27z00ktptrFz504DmHfffdfVdubMGTN48GBTqlQp4+fnZypUqGBee+01Y7fb3f6sV5PZcDN79mxTp04dExAQYG644QZz//33m8OHD6fbr2rVqsbf39/cdNNNZt68eWmOkTHGAGbkyJGu5fj4eDN48GBTpkwZ4+fnZ4oUKWKioqLMpk2bjDHO4w+kel3e5uVjPm3atFT72LFjh+nSpYsJDw83AQEBplKlSub555+/6ufMKNycP3/eAKZly5ap2jP7ezp16pR54IEHTEhIiAkLCzM9evQwMTExaeq+/PvYs2ePad26tQkODjYdOnQwxhhjt9vN22+/bapVq2b8/f1NRESE6devnzl9+nSqfW3cuNG0bNnS3HDDDSYgIMCULVvW9O7dO1WfL7/80tSpU8cEBwebkJAQU716dTNhwgTX+ytXrjSAWblyZar1MvN3cPkzHD582HTo0MEUKFDAhIeHm6eeeuqa//8quV/mx2hFcpnLpxMKFSrkalu7di1nzpxh8ODBGZ6i6NGjB9OmTePbb7/l1ltvZe3atZw+fZonnnjimk9DLFy4EIAHH3zwmtb/L9OmTSMhIYF+/frh7+9P8eLFadq0KbNnz2bkyJGp+s6aNQtvb2+6dOkCwMWLF2natClHjhyhf//+lC5dmp9++onnnnuOY8eOZfm8k3+7fJrxlltuYezYsRw/fpx33nmHdevWsWXLFgoWLAjAokWL6NatGzVq1GDs2LGcOXOGhx56iJIlS/7nPh555BHmzp3LoEGDqFatGn///Tdr165lx44d1KlThxdeeIG4uDgOHz7M22+/DXDVuUK//fYbTZo0wdfXl379+lG2bFn27t3LN998w6uvvur2MUjvbzezvyeHw0H79u3ZsGEDjz76KFWqVOHrr7+mZ8+e6e4rJSWF6OhobrvtNt566y2CgoIA6N+/v+t38fjjj7N//37ee+89tmzZwrp16/D19eXEiRO0bNmSIkWKMGzYMAoWLMiBAweYN2+ea/vLli3jvvvuo0WLFrz++usA7Nixg3Xr1jF48OAMj0Fm/w7Aeao5OjqaBg0a8NZbb7F8+XLGjRtHhQoVePTRR90+/pIHWJ2uRK7X5ZGb5cuXm5MnT5pDhw6ZuXPnmiJFihh/f39z6NAhV98JEyYYwMyfPz/D7Z0+fdoAplOnTsYYY955553/XOe/3H333QYwZ86cyVR/d0duQkNDzYkTJ1L1/fDDDw1gtm3blqq9WrVqpnnz5q7ll19+2RQoUMDs3r07Vb9hw4YZb29vc/DgwUzVnBn/NXKTlJRkIiIiTPXq1c2lS5dc7ZdH4UaMGOFqq1GjhilVqpQ5d+6cq23VqlWpRlku418jN2FhYWbgwIFXrTWj01LpjdzcfvvtJiQkxPz111+p+qZ3CjO9bY0aNcqcPHnSxMbGmjVr1phbbrnFAGbOnDmuvpn9PX311VcGSDUyYrfbTfPmzdMduQHMsGHDUm1zzZo1BjCff/55qvYlS5akap8/f366o6b/NHjwYBMaGnrVUZR/j9y483dw+TOMHj061TZr165t6tatm+E+JW/TzC3JM6KioihSpAiRkZF07tyZAgUKsHDhQkqVKuXqc+7cOQBCQkIy3M7l9+Lj41P979XW+S+e2MbV3HPPPRQpUiRVW6dOnfDx8WHWrFmutt9//53t27fTrVs3V9ucOXNo0qQJhQoV4tSpU65XVFQUdrudH3/8MUtqTs+vv/7KiRMnGDBgQKpJ4G3btqVKlSosWrQIgKNHj7Jt2zZ69OiRakSladOmmZqAW7BgQX755ReOHj163TWfPHmSH3/8kT59+lC6dOlU79lstkxtY+TIkRQpUoRixYrRpEkTduzYwbhx4+jcubOrT2Z/T0uWLMHX15e+ffu61vXy8mLgwIEZ7v/foxtz5swhLCyMO++8M9W+6tatS3BwMCtXrgRwjZ58++23JCcnp7vtggULcuHCBZYtW5apYwGZ/zv4p0ceeSTVcpMmTdi3b1+m9yl5i8KN5BkTJ05k2bJlzJ07lzZt2nDq1Cn8/f1T9bkcLi6HnPT8OwCFhob+5zr/xRPbuJpy5cqlaQsPD6dFixbMnj3b1TZr1ix8fHzo1KmTq+3PP/9kyZIlFClSJNUrKioKgBMnTmS437i4OGJjY12v06dPX9fn+OuvvwCoXLlymveqVKniev/y/1asWDFNv/Ta/u2NN97g999/JzIykvr16/PSSy9d8xfh5fWqV69+TesD9OvXj2XLlvHNN9/w5JNPcunSJex2e6o+mf09/fXXXxQvXtx1eumyjI6Lj49Pqn8AXN5XXFwcERERafZ3/vx5176aNm3KPffcw6hRowgPD6dDhw5MmzaNxMRE17YGDBhApUqVaN26NaVKlaJPnz4sWbLkqscjs38HlwUEBKQJ94UKFeLMmTNX3Y/kXZpzI3lG/fr1qVevHgAdO3bktttuo3v37uzatcv1r/uqVasCzjkSHTt2THc7v/32GwDVqlUDnP8xBdi2bVuG6/yXf26jSZMm/9nfZrNh0rlLw7+/8C4LDAxMt/3ee++ld+/exMTEUKtWLWbPnk2LFi0IDw939XE4HNx5550888wz6W6jUqVKGdY5ePBgPvnkE9dy06ZNc8WN2Lp27UqTJk2YP38+33//PW+++Savv/468+bNo3Xr1tlez4033ugKKe3atcPb25thw4bRrFkz19/09fyersbf3z/N5dcOh4OIiAg+//zzdNe5HCRsNhtz587l559/5ptvvmHp0qX06dOHcePG8fPPPxMcHExERAQxMTEsXbqU7777ju+++45p06bRo0ePVH871+N6L8mXvEcjN5IneXt7M3bsWI4ePcp7773nar/tttsoWLAgX3zxRYZBYcaMGYDzS+byOoUKFeLLL7/McJ3/0r59ewA+++yzTPUvVKgQZ8+eTdP+73+x/peOHTvi5+fHrFmziImJYffu3dx7772p+lSoUIHz588TFRWV7uvfp1r+6ZlnnmHZsmWu17hx49yq79/KlCkDwK5du9K8t2vXLtf7l/93z549afql15ae4sWLM2DAABYsWMD+/fu54YYbUk3+zewppfLlywPOU36e8sILLxASEsKLL77oasvs76lMmTIcO3aMixcvptpmZo/L5X39/fffNG7cON191axZM1X/W2+9lVdffZVff/2Vzz//nD/++IOZM2e63vfz86N9+/a8//777N27l/79+zNjxowMa8rs34FIRhRuJM+64447qF+/PhMmTCAhIQGAoKAghg4dyq5du3jhhRfSrLNo0SKmT59OdHQ0t956q2udZ599lh07dvDss8+mO6Ly2WefsWHDhgxradiwIa1atWLKlCksWLAgzftJSUkMHTrUtVyhQgV27tzJyZMnXW1bt25l3bp1mf784JzvEB0dzezZs5k5cyZ+fn5pRp+6du3K+vXrWbp0aZr1z549S0pKSobbr1atWqovvbp167pV37/Vq1ePiIgIJk2alOrUxnfffceOHTto27YtACVKlKB69erMmDEj1Q0BV69ezbZt2666D7vdTlxcXKq2iIgISpQokWqfBQoUSNMvPUWKFOH2229n6tSpHDx4MNV76f2tZEbBggXp378/S5cuJSYmBsj87yk6Oprk5GQmT57set/hcDBx4sRM779r167Y7XZefvnlNO+lpKS4gveZM2fSfMZatWoBuI7l33//nep9Ly8vbr755lR9/i2zfwciGdFpKcnTnn76abp06cL06dNdEw6HDRvGli1beP3111m/fj333HMPgYGBrF27ls8++4yqVaumGS5/+umn+eOPPxg3bhwrV6503aE4NjaWBQsWsGHDBn766aer1jJjxgxatmxJp06daN++PS1atKBAgQL8+eefzJw5k2PHjvHWW28B0KdPH8aPH090dDQPPfQQJ06cYNKkSdx0002uycmZ1a1bNx544AHef/99oqOjU11Ce/mzLVy4kHbt2tGrVy/q1q3LhQsX2LZtG3PnzuXAgQOpTmNdr+TkZF555ZU07YULF2bAgAG8/vrr9O7dm6ZNm3Lfffe5LgEuW7YsTz75pKv/mDFj6NChA40bN6Z3796cOXOG9957j+rVq1/1Dsjnzp2jVKlSdO7cmZo1axIcHMzy5cvZuHFjqpGnunXrMmvWLIYMGcItt9xCcHCwawTu3/73v/9x2223UadOHfr160e5cuU4cOAAixYtcoUTdw0ePJgJEybw2muvMXPmzEz/njp27Ej9+vV56qmn2LNnD1WqVGHhwoWu+VCZGZFq2rQp/fv3Z+zYscTExNCyZUt8fX35888/mTNnDu+88w6dO3fmk08+4f333+fuu++mQoUKnDt3jsmTJxMaGkqbNm0AePjhhzl9+jTNmzenVKlS/PXXX7z77rvUqlXLdZr433x9fTP9dyCSLmsv1hK5fhndxM8Y5yWwFSpUMBUqVEh1KardbjfTpk0zjRs3NqGhoSYgIMDcdNNNZtSoUVe9Q+vcuXNNy5YtTeHChY2Pj48pXry46datm1m1alWmar148aJ56623zC233GKCg4ONn5+fufHGG81jjz1m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xYoVo3z58gD88MMPfPvtt5w5c8Z1nN5//32WLVvGJ5984po0PGTIEJ588kmOHTtGoUKFOHDgAM8995xrO1nmP2flZLH69eubgQMHupbtdrspUaKEGTt2bLr9u3btatq2bZuqrUGDBqZ///6Z2l9WTiiOHva++XN4FWNGhhrzUkFjVr9hjN3u0f2IiOQkuXFCcXJysilatKgZN26c2bZtW6pXhQoVzAcffGCMcU6Q9fLyMl999VWq9ffu3WsAs3btWmOMMQcPHrzmCcWHDx82f/75Z4avAwcOZLju5QnFc+fOdbXt3LnzqhOK07N7927j5eVlli5dmmGfFStWGJvNZnbu3GmMcU4M9vLyMufOnUvVr1KlSmkmZv/T8OHDTWRkpElJSUn3fU9NKLY03CQmJhpvb28zf/78VO09evQwd911V7rrREZGumaoXzZixAhz8803p9s/ISHBxMXFuV6HDh3KsnDz1vN9jBkZauxvVjJm/1qPbl9EJCfKjeFm/vz5xs/Pz5w9ezbNe88880yqK3z69etnypYta77++muzb98+s3r1anPrrbeaW2+91TgcDle/iRMnGpvNZvr06WNWrVplDhw4YNauXWv69etnhgwZkmWf5ZFHHjGlS5c2P/zwg/n1119Nw4YNTcOGDVP1qVy5spk3b55refbs2WblypVm7969ZsGCBaZMmTKmU6dOqdaZOnWqWb9+vdmzZ4/59NNPTeHChVN9jpMnT5obbrjBdOrUycTExJhdu3aZoUOHGl9fXxMTE+Pq98Ybb5jffvvN/P7772b06NHG19c3zXf+P+WJcHPkyBEDmJ9++ilV+9NPP23q16+f7jq+vr7miy++SNU2ceJEExERkW7/kSNHGiDNKyvCTblnF5p3XuhhLpw+5tFti4jkVLkx3LRr1860adMm3fd++eUXA5itW7caY5yfb+TIkaZKlSomMDDQlCtXzvTr18+cPHkyzbrLli0z0dHRplChQiYgIMBUqVLFDB061Bw9ejTLPsulS5fMgAEDTKFChUxQUJC5++67zbFjqb+DADNt2jTX8jvvvGNKlSplfH19TenSpc2LL75oEhMTU63z7LPPmqJFixpfX19z4403mnHjxqUKc8YYs3HjRtOyZUtTuHBhExISYm699VazePHiVH2aNWtmwsLCTEBAgGnQoEGa99P7PJ4IN7b//+CW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YiLu01PBRfKoVq1aMW3atFRtRYoUsaia1EJDQ9m1axcOh4OtW7fSu3dvjh49ytKlS6972//19HiAsLCw695PZtx0000sX74cu93Ojh076NOnD3FxccyaNStb9i+SX2nkRiSP8vf3p1ixYqle3t7ejB8/nho1alCgQAEiIyMZMGAA58+fz3A7W7dupVmzZoSEhBAaGkrdunX59ddfXe+vXbuWJk2aEBgYSGRkJI8//jgXLly4am02m41ixYpRokQJWrduzeOPP87y5cu5dOkSDoeD0aNHU6pUKfz9/alVqxZLlixxrZuUlMSgQYMoXrw4AQEBlClThrFjx6ba9uXTUuXKlQOgdu3a2Gw27rjjDiD1aMhHH31EiRIlUj2FG6BDhw706dPHtfz1119Tp04dAgICKF++PKNGjSIlJeWqn9PHx4dixYpRsmRJoqKi6NKlC8uWLXO9b7fbeeihhyhXrhyBgYFUrlyZd955x/X+Sy+9xCeffMLXX3/tGgVatWoVAIcOHaJr164ULFiQwoUL06FDBw4cOHDVekTyC4UbkXzGy8uL//3vf/zxxx988skn/PDDDzzzzDMZ9r///vspVaoUGzduZNOmTQwbNgxfX18A9u7dS6tWrbjnnnv47bffmDVrFmvXrmXQoEFu1RQYGIjD4SAlJYV33nmHcePG8dZbb/Hbb78RHR3NXXfdxZ9//gnA//73PxYuXMjs2bPZtWsXn3/+OWXLlk13uxs2bABg+fLlHDt2jHnz5qXp06VLF/7++29Wrlzpajt9+jRLlizh/vvvB2DNmjX06NGDwYMHs337dj788EOmT5/Oq6++munPeODAAZYuXYqfn5+rzeFwUKpUKebMmcP27dsZMWIEzz//PLNnzwZg6NChdO3alVatWnHs2DGOHTtGo0aNSE5OJjo6mpCQENasWcO6desIDg6mVatWJCUlZbomkTwryx/NKSLZrmfPnsbb29sUKFDA9ercuXO6fefMmWNuuOEG1/K0adNMWFiYazkkJMRMnz493XUfeugh069fv1Rta9asMV5eXubSpUvprvPv7e/evdtUqlTJ1KtXzxhjTIkSJcyrr76aap1bbrnFDBgwwBhjzGOPPWaaN29uHA5HutsHzPz5840xxuzfv98AZsuWLan6/PuJ5h06dDB9+vRxLX/44YemRIkSxm63G2OMadGihRkz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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Add Top Features"
      ],
      "metadata": {
        "id": "Bqx2pey60qPX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "feature_names = tfidf.get_feature_names_out()\n",
        "coefficients = best_lr.coef_[0]\n",
        "\n",
        "top_real = np.argsort(coefficients)[-10:]\n",
        "top_fake = np.argsort(coefficients)[:10]\n",
        "\n",
        "print(\"\\nTop words for REAL news:\\n\")\n",
        "for i in reversed(top_real):\n",
        "    print(feature_names[i])\n",
        "\n",
        "print(\"\\nTop words for FAKE news:\\n\")\n",
        "for i in top_fake:\n",
        "    print(feature_names[i])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N7jc1Pug0rUG",
        "outputId": "a9fcb8ec-8d3d-4068-b3ed-03413181eca6"
      },
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Top words for REAL news:\n",
            "\n",
            "president donald\n",
            "ve\n",
            "don\n",
            "president barack\n",
            "thursday\n",
            "republican\n",
            "statement\n",
            "nov\n",
            "spokesman\n",
            "comment\n",
            "\n",
            "Top words for FAKE news:\n",
            "\n",
            "video\n",
            "image\n",
            "gop\n",
            "president trump\n",
            "breaking\n",
            "hillary\n",
            "images\n",
            "wire\n",
            "watch\n",
            "president obama\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# MODEL SAVING"
      ],
      "metadata": {
        "id": "NKDnvd4446jN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle\n",
        "\n",
        "with open('fake_news_model.pkl', 'wb') as f:\n",
        "    pickle.dump(best_lr, f)\n",
        "\n",
        "with open('tfidf_vectorizer.pkl', 'wb') as f:\n",
        "    pickle.dump(tfidf, f)\n",
        "\n",
        "print(\"Model and vectorizer saved successfully!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cDAz3i1O48a9",
        "outputId": "d6972602-085a-43fb-cebb-a8cf10dfb875"
      },
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model and vectorizer saved successfully!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Load Model & Vectorizer"
      ],
      "metadata": {
        "id": "KO48XBPh4-zU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "with open('fake_news_model.pkl', 'rb') as f:\n",
        "    loaded_model = pickle.load(f)\n",
        "\n",
        "with open('tfidf_vectorizer.pkl', 'rb') as f:\n",
        "    loaded_tfidf = pickle.load(f)\n",
        "\n",
        "print(\"Model loaded successfully!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0r9guy_N5CDN",
        "outputId": "28de0fa8-3977-4ced-b389-b1a0b63fb168"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model loaded successfully!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# FINAL PREDICTION FUNCTION"
      ],
      "metadata": {
        "id": "NH1s8sSv5Fql"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_news_final(text, threshold=0.4):\n",
        "    processed = preprocess_text(text)\n",
        "    processed = remove_bias_words(processed)\n",
        "\n",
        "    vectorized = loaded_tfidf.transform([processed])\n",
        "    prob = loaded_model.predict_proba(vectorized)[0][1]\n",
        "\n",
        "    if prob >= threshold:\n",
        "        label = \"Real News ✅\"\n",
        "    else:\n",
        "        label = \"Fake News ❌\"\n",
        "\n",
        "    return f\"{label} (Confidence: {prob:.2f})\""
      ],
      "metadata": {
        "id": "tmzl6XC85Gre"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Final Testing"
      ],
      "metadata": {
        "id": "BYl47Jxf5J8T"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "samples = [\n",
        "    \"Government launches new healthcare policy to improve hospitals\",\n",
        "    \"Aliens have landed in New York and taken control\",\n",
        "    \"Stock market rises after strong economic growth report\",\n",
        "    \"Breaking: secret conspiracy discovered in government\"\n",
        "]\n",
        "\n",
        "for s in samples:\n",
        "    print(f\"\\nText: {s}\")\n",
        "    print(\"Prediction:\", predict_news_final(s))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GkmPW1NE5Nn3",
        "outputId": "f8f76909-1bd2-4f9c-cd50-00bb05d87994"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Text: Government launches new healthcare policy to improve hospitals\n",
            "Prediction: Real News ✅ (Confidence: 0.79)\n",
            "\n",
            "Text: Aliens have landed in New York and taken control\n",
            "Prediction: Fake News ❌ (Confidence: 0.24)\n",
            "\n",
            "Text: Stock market rises after strong economic growth report\n",
            "Prediction: Real News ✅ (Confidence: 0.64)\n",
            "\n",
            "Text: Breaking: secret conspiracy discovered in government\n",
            "Prediction: Fake News ❌ (Confidence: 0.00)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Final Model Summary\n"
      ],
      "metadata": {
        "id": "etxUoYc-5P5d"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"FINAL MODEL PERFORMANCE SUMMARY\\n\")\n",
        "\n",
        "print(f\"Naive Bayes F1 Score: {nb_results[3]:.4f}\")\n",
        "print(f\"Logistic Regression F1 Score: {lr_results[3]:.4f}\")\n",
        "print(f\"Cross-validation F1 Score: {cv_scores.mean():.4f}\")\n",
        "\n",
        "print(\"\\nBest Model: Logistic Regression (Tuned)\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Vaob8TOH5QZU",
        "outputId": "36e34ee8-3efd-42fb-acf5-c98f872bd336"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "FINAL MODEL PERFORMANCE SUMMARY\n",
            "\n",
            "Naive Bayes F1 Score: 0.9317\n",
            "Logistic Regression F1 Score: 0.9862\n",
            "Cross-validation F1 Score: 0.9856\n",
            "\n",
            "Best Model: Logistic Regression (Tuned)\n"
          ]
        }
      ]
    }
  ]
}