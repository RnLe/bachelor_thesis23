\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{solutionclass}[2023/03/03 My Custom LaTeX Class for exercise solutions]

\LoadClass[a4paper, twoside, 10pt]{article}

% LANGUAGE AND ENCODING
\RequirePackage[german]{babel} % Allows to use multiple languages in the document
\RequirePackage[utf8]{inputenc} % Defines the input encoding
\RequirePackage[T1]{fontenc} % Defines the font encoding

% TEXT FORMATTING AND MANIPULATION
\setlength{\parindent}{0pt}
%\RequirePackage{indentfirst} % Indent the first paragraph of a section
\RequirePackage{datetime} % Provides various features of date and time formatting
\RequirePackage{csquotes} % Context sensitive quotation facilities
\RequirePackage{mathrsfs} % Provides additional math scripts fonts
\RequirePackage{titlesec} % Allows customization of section titles
\RequirePackage{upgreek} % Provides upright greek letters
\RequirePackage{dsfont} % Provides various math symbols such as the indicator function
\RequirePackage{pifont} % Provides various symbols such as check marks and crosses
\RequirePackage{lipsum} % Generates Lorem Ipsum text for testing purposes

% MATHEMATICS AND PHYSICS
\RequirePackage{amsmath} % for mathematical symbols
\RequirePackage{amssymb} % for mathematical symbols
\RequirePackage{mathtools} % for mathematical typesetting
\RequirePackage{physics} % for mathematical physics
\RequirePackage{stmaryrd} % for mathematical symbols
\RequirePackage{tensor} % for tensors
\RequirePackage{calc} % for mathematical calculations

% FIGURES, TIKZ AND BOXES
\RequirePackage{graphicx} % for including images
\RequirePackage{float} % provides H option for figures and tables
\RequirePackage{xcolor} % for color support
\RequirePackage{tikz} % for creating graphics and diagrams
\RequirePackage{tikz-3dplot} % for 3D graphics in TikZ
\RequirePackage{pgfplots} % for creating plots and diagrams
\RequirePackage[most]{tcolorbox} % for creating colored boxes and frames

% PAGE AND TEXT STYLES

\RequirePackage[top = 2.5cm,
                bottom = 3.5cm,
                left = 2.5cm,
                right = 2.5cm,
                ]{geometry} % Page margins
\RequirePackage[shortlabels]{enumitem} % Customized lists
\RequirePackage{microtype} % Improves typography
\RequirePackage{lastpage} % Reference to the last page
\RequirePackage{etoolbox} % Toolbox of programming tools
\RequirePackage{multicol} % Multiple columns
\RequirePackage{caption} % Customizing captions
\renewcommand{\familydefault}{\sfdefault} % Set default font family to sans serif
\RequirePackage{lmodern} % Latin Modern fonts
\RequirePackage{nameref} % Cross-referencing by name
\RequirePackage{changepage} % Change page dimensions
\RequirePackage{listings}   % Format code and implement code-files
\RequirePackage{dirtree}    % Render folder structures
\RequirePackage{pdfpages}
\RequirePackage{wrapfig}
\RequirePackage{subfigure}

% BIBLIOGRAFY
\RequirePackage[backend    = biber,
                sorting    = none,
                backref    = false,
                uniquename = true,
                doi        = true,
                autolang   = hyphen,
                maxbibnames= 3,
                giveninits = true]{biblatex} % Custom bibliography
                % Options articlein/dashed removed: not supported by all biblatex versions

\DefineBibliographyStrings{german}{%
    andothers = {et al.}, % statt 'u.a.'
}

\RequirePackage[colorlinks = true,
                urlcolor = LinkColor,
                anchorcolor = LinkColor,
                citecolor = LinkColor,
                filecolor = LinkColor,
                linkcolor = LinkColor,
                menucolor = LinkColor,
                linktoc = section,
                bookmarks = true,
                pdfusetitle]{hyperref} % Hyperlinks
\RequirePackage[
  locale=DE,
  separate-uncertainty=true,
  per-mode=symbol-or-fraction,
]{siunitx}

\RequirePackage{fancyhdr} % Custom headers and footers

\pagestyle{fancy}
\fancyhf{}
\rhead{\thepage}
\lhead{\nouppercase{\leftmark}}
\renewcommand{\headrulewidth}{0.4pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% BIBLIOGRAFYU SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The below settings add the .bib archive and define the format which will be compilated
\addbibresource{bib.bib} %Load references
\DeclareFieldFormat[article,periodical]{volume}{\mkbibbold{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The below commands was created to make a onte kind of reference citing
\newcommand{\therefs}{%
    {%\bfseries%
    \newline%
    \textcolor{myLColor}{References:}%
    }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtEndDocument{%
    
} 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% TIKZ SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usetikzlibrary{%
    decorations.pathreplacing, % Create decorative path elements that replace parts of the original path.
    decorations.pathmorphing, % Create decorative path elements that morph the original path.
    decorations.markings, % Place decorations along a path.
    shapes.multipart, % Create nodes with multiple parts.
    shapes.geometric, % Provide additional geometric shapes.
    arrows.meta, % Define arrowheads for paths.
    tikzmark, % Place markers on a path or in a TikZ picture.
    fadings, % Create gradient and transparency effects.
    arrows, % Provide additional arrowheads.
    angles, % Measure and display angles between lines.
    quotes, % Place quotes on a path.
    calc, % Provide additional mathematical functions.
    3d, % Create three-dimensional TikZ diagrams.
}
\pgfplotsset{compat=1.18} % Sets the compatibility level for the pgfplots package to version 1.18, ensuring consistent compilation of plots and diagrams in the document.

\tikzset{>=stealth} % Arrow style for all tikZ environment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% COLOR SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To change the darker color of the document, like the cover page background, change the hexadecimal code #101010 to the color which you want. To know what hexadecimal code you need, access https://htmlcolorcodes.com
\definecolor{Grey}{HTML}{c1c1c1} % Grey
\definecolor{myDColor}{HTML}{101010} % Darker Color
\definecolor{TUGreen}{HTML}{84B819} % TU Green
\definecolor{DTUGreen}{HTML}{678f13} % TU Green in dark
% To change the lighter color of the document, like the box solution backaground, change the hexadecimal code #e93820 to the color which you want
\definecolor{myLColor}{HTML}{7736d9} % Lighter Color
% To change the link color of the document, like the mentions, change the hexadecimal code #eb5a00 to the color which you want
\definecolor{LinkColor}{HTML}{678f13} % Link Color
\definecolor{TUCLight}{HTML}{9e82d8} % Light complementary color respective to TUGreen

% colors for c++
\definecolor{comments}{HTML}{6A9955}
\definecolor{keywords}{HTML}{C586C0}
\definecolor{string}{HTML}{CE9178}

\definecolor{matplot_C0}{HTML}{1F77B4}



% Define the style for the listings package (code formatting)
\lstdefinestyle{cpp}{
    backgroundcolor=\color{myLColor!5},   
    commentstyle=\color{comments},
    keywordstyle=\color{keywords},
    numberstyle=\tiny\color{Grey},
    stringstyle=\color{string},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

%\lstset{style=cpp}

% Custom table formatting
\DeclareCaptionFormat{tableformat}{\textbf{\textcolor{DTUGreen}{#1#2}}#3}
\captionsetup[table]{format=tableformat, width=0.9\textwidth}

% Custom figure formatting
\DeclareCaptionFormat{figureformat}{\textbf{\textcolor{DTUGreen}{#1#2}}#3}
\captionsetup[figure]{format=figureformat, width=0.9\textwidth}

% CUSTOM COMMANDS
% custom command to highlight nodes in the tree structure
\newcommand{\hlnode}[1]{\DTstyle{\textcolor{TUGreen}{#1}}}

\renewcommand{\texttt}[1]{\textcolor{TUCLight}{\ttfamily #1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This command create a polygon on the left side of the section title
\newcommand{\polygonSection}{%
\begin{tikzpicture}[remember picture, overlay, baseline=1.5ex]
\node[%
    regular polygon sides   = 6, 
    rounded corners,
    regular polygon,
    minimum size            = 1.58cm,
    ultra thick,
    inner sep               = 0pt,
    fill                    = myDColor!85,
    anchor                  = south
] at (current page.north west |- 0,0) (red polygon) {};
\foreach \i in {2.5,...,4.5}{%
    \node[%
        TUGreen,
        regular polygon sides   = 6, 
        regular polygon,
        rounded corners,
        minimum size            = \i cm,
        ultra thick,
        draw,
    ] at (red polygon) {};
}
\end{tikzpicture}}

\RequirePackage{titlesec}
\titleformat{\section}
    {\vspace{2em}\normalfont\bfseries\color{myDColor}\huge}
    {\thesection \hspace{0.5em}}{5pt}
    {\polygonSection}
\titlespacing{\section}{0pt}{5pt}{5pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This command change the color of the label number in \label{} commands
\makeatletter
\def\tagform@#1{\maketag@@@{\normalfont(\color{myLColor}\ignorespaces#1\unskip\@@italiccorr\color{black})}}
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DOCUMENT COVER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here the commands for building the document cover were made.
\newcommand{\pretitle}[9]{%
\begin{titlepage}
\pagestyle{empty}
\begin{tikzpicture}[remember picture,overlay]
    % BACKGROUND
    \fill[myDColor] (current page.south west) rectangle (current page.north east);
    \foreach \i in {2.5,...,22}{%
        \node[%
            myDColor!80,
            regular polygon sides   = 6, 
            regular polygon,
            rounded corners,
            minimum size            = \i cm,
            ultra thick,
            draw,
        ] at ($(current page.west)+(4,-5)$) {};
    }
    % BACKGROUND POLYGONS
    \foreach \i in {0.5,...,20}{%
        \node[%
            TUGreen!85,        
            regular polygon sides   = 6, 
            regular polygon,
            rounded corners,
            minimum size            = \i cm,
            ultra thick,
            draw,
        ] at ($(current page.north west)+(2.5,0)$) {};
    }
    \foreach \i in {0.5,...,22}{%
        \node[%
            myDColor!90,        
            regular polygon sides   = 6, 
            regular polygon,
            rounded corners,
            minimum size            = \i cm,
            ultra thick,
            draw,
        ] at ($(current page.north east)+(0,-10)$) {};
    }
    \foreach \i in {19,...,6}{%
        \node[
            TUGreen!85,        
            regular polygon sides   = 6, 
            regular polygon,
            rounded corners,
            minimum size            = \i cm,
            ultra thick,
            draw,
        ] at ($(current page.south east)+(-0.2,0.1)$) {};
    }
    % Pretext
    \node[%
        myLColor!5,
        rounded corners,        
        minimum height  = 2cm,         
        minimum width   = 0.625*\paperwidth,
    ] (title) at ($(current page.north)+(0,-9)$){%
        {\fontsize{20}{20} \selectfont \textit{#1}}
    };
    % Pretext 2nd line
    \node[%
        myLColor!5,
        rounded corners,        
        minimum height  = 2cm,         
        minimum width   = 0.625*\paperwidth,
    ] (title) at ($(current page.north)+(0,-10)$){%
        {\fontsize{20}{20} \selectfont \textit{#2}}
    };
    % Title
    \node[%
        TUGreen!10,
        rounded corners,
        minimum height  = 4cm, 
        minimum width   = 0.625*\paperwidth,
    ] at ($(current page.north)+(0,-12)$){%
        {\fontsize{23}{23} \bfseries #3}
    };
    % Name
    \node[%
        TUGreen!5,
        rounded corners,
        minimum height  = 2cm, 
        minimum width   = 0.625*\paperwidth,
    ] at ($(current page.north)+(0,-14)$){%
        {\Large \textsc{#4}}
    };
    % Born
    \node[%
        TUGreen!5,
        rounded corners,
        minimum height  = 1cm, 
        minimum width   = 0.625*\paperwidth,
    ] at ($(current page.north)+(0,-15)$){%
        {\fontsize{23}{23} #5}
    };
    % Year
    \node[%
        TUGreen!5,
        rounded corners,
        minimum height  = 4cm, 
        minimum width   = 0.625*\paperwidth,
    ] at ($(current page.north)+(0,-17)$){%
        {\fontsize{21}{21} \bfseries #6}
    };
    % Group
    \node[%
        TUGreen!5,
        rounded corners,
        minimum height  = 2cm, 
        minimum width   = 0.625*\paperwidth,
        align=left,
    ] at ($(current page.north)+(0,-21)$){%
        {\Large #7}
    };
    % Faculty
    \node[%
        TUGreen!5,
        rounded corners,
        minimum height  = 2cm, 
        minimum width   = 0.625*\paperwidth,
        align=left,
    ] at ($(current page.north)+(0,-22)$){%
        {\Large #8}
    };
    % University
    \node[%
        TUGreen!5,
        rounded corners,
        minimum height  = 2cm, 
        minimum width   = 0.625*\paperwidth,
        align=left,
    ] at ($(current page.north)+(0,-23)$){%
        {\Large #9}
    };
    % Institute name
    \node[text = myLColor!5] (Institute) at ($(title.north east)+(-1,3)$$) {};
    % University logo
    %\node[] at ($(Institute.north)+(0,0.6)$){\includegraphics[width=7cm]{PhysicsLogo.png}};
    %\node[] at ($(Institute.north)+(0,3)$){\includegraphics[width=7cm]{TUDoLogo.png}};
    
\end{tikzpicture}
\end{titlepage}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NEW ENVIRONMENTS AND NEW COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This environment was made to produce a box which maintain the width and height of the text inside it.
\newtcolorbox{task}[1][]{%
    enhanced,
    skin first=enhanced,
    skin middle=enhanced,
    skin last=enhanced,
    before upper={\parindent0pt},
    breakable,
    boxrule = 0pt,
    frame hidden,
    borderline west = {4pt}{0pt}{myDColor},
    colback = myLColor!5,
    coltitle = myLColor!5,
    sharp corners,
    rounded corners = southeast,
    arc is angular,
    arc = 3mm,
    attach boxed title to top left,
    boxed title style = {%
        enhanced,
        colback = myDColor,
        colframe = myDColor,
        top = 0pt,
        bottom = 0pt,
        sharp corners,
        rounded corners = northeast,
        arc is angular,
        arc = 2mm,
        rightrule = 0pt,
        bottomrule = 0pt,
        toprule = 0pt,
    },
    title = {\bfseries #1}, % Fügt ein Label zur Box hinzu, das dem Titel entspricht
    overlay unbroken={%
        \node[anchor=west, color=black!70] at (title.east) {};
        \path[fill = tcbcolback!80!black] 
            ([yshift = 3mm]interior.south east) -- ++(-0.4,-0.1) -- ++(0.1,-0.2);
    },
    overlay first = {%
        \node[anchor=west, color=black!70] at (title.east) {};
        \path[fill = tcbcolback!80!black] 
            ([yshift = 3mm]interior.south east) -- ++(-0.4,-0.1) -- ++(0.1,-0.2);
    },
    overlay middle={%
        \path[fill = tcbcolback!80!black] 
            ([yshift = -3mm]interior.north east) -- ++(-0.4,0.1) -- ++(0.1,0.2);
        \path[fill = tcbcolback!80!black] 
            ([yshift = 3mm]interior.south east) -- ++(-0.4,-0.1) -- ++(0.1,-0.2);
    },
    overlay last={%
        \path[fill = tcbcolback!80!black] 
            ([yshift = -3mm]interior.north east) -- ++(-0.4,0.1) -- ++(0.1,0.2);
        \path[fill = tcbcolback!80!black] 
            ([yshift = 3mm]interior.south east) -- ++(-0.4,-0.1) -- ++(0.1,-0.2);
    },
    extras middle and last = { rounded corners = northeast }
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The below commands was made to personalize the align and align* environments to have a orange line between to equalities.
\newcommand*\eqlineset{\pgfqkeys{/eqlines}}
\eqlineset{
  picture/.style={baseline, remember picture},
  line/.style={thick, draw=myLColor, line cap=round},
  node/.style={
    minimum size=+0pt, inner sep=+0pt, outer ysep=+.3333em, anchor=base,
    name=tma\thetikzmarkalign},
}
\newcounter{tikzmarkalign}
\makeatletter
\AtBeginEnvironment{align}{%
  \setcounter{tikzmarkalign}{0}%
  \newcommand*\noeq{\hphantom{={}}}%
  \newcommand*\eq{
    \ifnum\value{tikzmarkalign}=0
      \mathrel{\tikz[/eqlines/picture]\node[/eqlines/node]{$=$};}
    \else
      \mathrel{
        \begin{tikzpicture}[/eqlines/picture]
          \node[/eqlines/node] (tma\thetikzmarkalign) {$=$};
          \unless\ifmeasuring@
            \path[/eqlines/line, overlay] (tma\thetikzmarkalign)
                        -- (tma\pgfinteval{\thetikzmarkalign-1});
          \fi
        \end{tikzpicture}
      }
    \fi
    \stepcounter{tikzmarkalign}
  }%
}
\makeatother
\makeatletter
\AtBeginEnvironment{align*}{%
  \setcounter{tikzmarkalign}{0}%
  \newcommand*\noeq{\hphantom{={}}}%
  \newcommand*\eq{
    \ifnum\value{tikzmarkalign}=0
      \mathrel{\tikz[/eqlines/picture]\node[/eqlines/node]{$=$};}
    \else
      \mathrel{
        \begin{tikzpicture}[/eqlines/picture]
          \node[/eqlines/node] (tma\thetikzmarkalign) {$=$};
          \unless\ifmeasuring@
            \path[/eqlines/line, overlay] (tma\thetikzmarkalign)
                        -- (tma\pgfinteval{\thetikzmarkalign-1});
          \fi
        \end{tikzpicture}
      }
    \fi
    \stepcounter{tikzmarkalign}
  }%
}
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This command make a simple line to divide texts. The width is adaptable to the text width.
\newcommand{\divider}{\setlength{\parindent}{0pt}\textcolor{myDColor!30}{\rule{\linewidth}{0.6pt}}\smallskip\setlength{\parindent}{0pt}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This command was made to create a box (which can be labeled) to write equations which are answers for a random problem
\NewEnviron{answer}{
\begin{equation}
\tcbhighmath[
         boxrule = 1pt,
         colframe = TUGreen,
         colback = black!70,
         sharp corners,
         rounded corners = southeast,
         rounded corners = northeast,
         rounded corners = southwest,
         rounded corners = northwest,
         arc is angular,
         arc = 3mm,
         coltext = white,
     ]{\BODY}\end{equation}}

% This command is almost identical to the one above, but its used to highlight inline math
\newcommand{\highlight}[1]{%
    \tcbhighmath[
        boxrule = 1pt,
        colframe = TUGreen,
        colback = black!60,
        sharp corners,
        rounded corners = southeast,
        rounded corners = northeast,
        rounded corners = southwest,
        rounded corners = northwest,
        arc is angular,
        arc = 2mm,
        coltext = white,
        boxsep = 1pt,
    ]{#1}%
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This renewcommand was created to change the arrow notation from vectors to the bold notation of them.
\renewcommand\vec{\boldsymbol}
% This command removes the italic from variables to display them as constants
\newcommand{\const}[1]{\mathrm{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This command was created to make comments inside the text using the a more beautifull " ".
\newcommand{\Acomment}[1]{\textquotedblleft \textit{#1}\textquotedblright }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\firstcorrector}[1]{\def\thefirstcorrector{#1}}
\newcommand{\secondcorrector}[1]{\def\thesecondcorrector{#1}}
\newcommand{\submissiondate}[1]{\def\thesubmissiondate{#1}}

\newcommand{\makecorrectorpage}{%
  \thispagestyle{empty}
  \vspace*{\fill}
  \begin{tabular}{@{}l@{\hspace{2em}}l@{}}
      Erstgutachter:  & \thefirstcorrector \\
      Zweitgutachter: & \thesecondcorrector \\
      Abgabedatum:    & \thesubmissiondate
  \end{tabular}
  \newpage
}

% To show "Literatur" instead of "References"
\renewcommand{\refname}{Literatur}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The below commands is to maintain the font to the symbol mathbb{1} different and more beauty
\DeclareMathAlphabet{\mathbbold}{U}{bbold}{m}{n}
\newcommand*{\boldone}{\mathbbold{1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFINE YOUR NEW COMMANDS FROM HERE:
