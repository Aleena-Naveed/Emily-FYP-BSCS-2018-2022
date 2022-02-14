import React, { useState } from "react";
import IconButton from '@material-ui/core/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";
import Button from '@material-ui/core/Button';
import ReplaySharpIcon from '@mui/icons-material/ReplaySharp';
import HighlightOffSharpIcon from '@mui/icons-material/HighlightOffSharp';
import SkipNextSharpIcon from '@mui/icons-material/SkipNextSharp';
import Webcam from "react-webcam";
import AudioReactRecorder, { RecordState } from 'audio-react-recorder'
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';



//Animations

import greeting from "../animations/intro.mp4";
import quest1 from "../animations/doing.mp4"
import quest2 from "../animations/feel down.mp4"
import quest3 from "../animations/introvert.mp4"
import quest4 from "../animations/relax.mp4"
import quest5 from "../animations/argue about.mp4"
import quest6 from "../animations/for fun.mp4"
import quest7 from "../animations/positive.mp4"
import quest8 from "../animations/any regret.mp4"
import quest9 from "../animations/memorable.mp4"
import quest10 from "../animations/sleep.mp4"
import quest11 from "../animations/felt happy.mp4"
import quest12 from "../animations/get annoyed.mp4"
import quest13 from "../animations/friend.mp4"
import quest14 from "../animations/thoughts.mp4"
import quest15 from "../animations/advise.mp4"
import phqintro from "../animations/phq intro.mp4"
import Image from "../animations/doing.png"
import { useHistory } from "react-router-dom";
import { drawMesh } from "../utilities";
import * as tf from "@tensorflow/tfjs";
import * as facemesh from "@tensorflow-models/face-landmarks-detection";
import axios from "axios";





const gen_quest = [
    // {
    //     questiontext: "How are you doing today?",
    //     src: quest1,
    //     section: true,
    // },
    // {
    //     questiontext: "Do you feel down ?",
    //     src: quest2,
    //     section: true,
    // },
    // {
    //     questiontext: "Do you consider yourself introvert ?",
    //     src: quest3,
    //     section: true,
    // },
    // {
    //     questiontext: "What do you do to relax ?",
    //     src: quest4,
    //     section: true,
    // },
    // {
    //     questiontext: "When was the last time you argued someone and what was it about ?",
    //     src: quest5,
    //     section: true,
    // },
    // {
    //     questiontext: "What are some things you like to do for fun ?",
    //     src: quest6,
    //     section: true,
    // },
    // {
    //     questiontext: "Who’s someone that’s been positive influence in your life ?",
    //     src: quest7,
    //     section: true,
    // },
    // {
    //     questiontext: "Is there anything you regret ?",
    //     src: quest8,
    //     section: true,
    // },
    // {
    //     questiontext: "What’s one of your most memorable moments ?",
    //     src: quest9,
    //     section: true,
    // },
    // {
    //     questiontext: "How easy is it for you to get good night’s sleep ?",
    //     src: quest10,
    //     section: true,
    // },
    // {
    //     questiontext: "When was the last time you felt really happy ?",
    //     src: quest11,
    //     section: true,
    // },
    // {
    //     questiontext: "What do you do when you are annoyed ?",
    //     src: quest12,
    //     section: true,
    // },
    // {
    //     questiontext: "How would your best friend describe you ?",
    //     src: quest13,
    //     section: true,
    // },
    {
        questiontext: "Have you noticed any changes in your thoughts and behavior lately ?",
        src: quest14,
        section: true,
    },
    {
        questiontext: "What advice would you give yourself ten or twenty years ago ?",
        src: quest15,
        section: true,
    },
    {
        questiontext: "I will be asking you questions regarding Patient Health Questionare, to measure your level of depression",
        src: phqintro,
        section: false,
    },


]


const questions = [
    {
        questionText: 'Over the last 2 weeks, how often you little interest or pleasure in doing things',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    },
    {
        questionText: 'Over the last 2 weeks, how often you been feeling down, depressed, or hopeless',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    }, {
        questionText: 'Over the last 2 weeks, how often you having trouble falling or staying asleep, or sleeping too much',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    }, {
        questionText: 'Over the last 2 weeks, how often you been feeling tired or having little energy',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    }, {
        questionText: 'Over the last 2 weeks, how often you having poor appetite or overeating',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    }, {
        questionText: 'Over the last 2 weeks, how often you been feeling bad about yourself or that you are a failure or have let yourself or your family down',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    }, {
        questionText: 'Over the last 2 weeks, how often you having trouble concentrating on things, such as reading the newspaper or watching television',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    }, {
        questionText: 'Over the last 2 weeks, how often you having Thoughts that you would be better off dead, or of hurting yourself',
        answerOptions: [
            { answerText: 'Not at all', score: 0 },
            { answerText: 'Several Days', score: 1 },
            { answerText: 'More than half of the days', score: 2 },
            { answerText: 'Nearly every day', score: 3 },
        ],
    },
];



const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,

        minHeight: "100vh",
        overflow: "hidden",
    },
    modal: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    paper: {
        backgroundColor: "#3585dargba(0, 181, 0, 0.12)",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        padding: theme.spacing(2, 4, 3),
    },

    videoDim: {
        position: 'absolute',
        right: '0',
        bottom: '0',
        minWidth: '100vw',
        height: '100vh',
        objectFit: "cover",
        zIndex: '-1',

    },
    BtnPlacement: {
        marginBottom: 5,
        marginRight: 5,

    },


    Btn: {

        borderRadius: 100,
        background: "#3585da",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        fontWeight: 'bold',
        color: "#fff",
        marginTop: 10,
        marginBottom: 5,
    },
    BtnContained: {
        "&:hover": {
            background: "rgba(0, 0, 0, 0.16)",
            color: "#3585da",
        }
    },
    BtnText: {
        fontWeight: 'bold',
        color: "#fff",
        fontSize: 16,
    },
    Wbcam: {
        width: "200px",
        height: "200px"
    },
    question_sect: {
        width: "100%",
        position: "relative"
    },
    question_count: {
        marginBottom: "10px",
        margin: "10px",
    },
    question_txt: {
        marginBottom: "12px",
        alignText: 'center'
    },
    answer_sect: {
        width: "100%",
        color: "#fff",

    },
    quest_grid: {
        color: "#fff",
        marginLeft: '10px',
        marginRight: '10px',
        borderRadius: 0,
        background: "rgba(0, 0, 0, 0.35)",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        borderRadius: 5,

    },

    Btn_quest: {

        borderRadius: 5,
        background: "rgba(53, 133, 218, 0.2)",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        color: "#fff",
        marginTop: 10,
        marginBottom: 5,
        marginRight: 5,
    },
    Btn_questContained: {
        "&:hover": {
            background: "rgba(0, 0, 0, 0.16)",

        }
    },
    Btn_questText: {
        color: "#fff",
        fontSize: 16,
    },
    paperContainer: {
        backgroundImage: `url(${Image})`
    }

}))








export default function Profile() {
    const classes = useStyles();
    const history = useHistory();
    const vidref = React.useRef();

    const webcamref = React.useRef();
    const canvasRef = React.useRef();

    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [Totalscore, setTotalScore] = useState(0);
    const [currentVideo, setCurrentVideo] = useState(0);
    const [sect1, setSect1] = useState(false);
    const [sect2, setSect2] = useState(false)
    const [sect3, setSect3] = useState(true)
    const [record, setRecord] = useState(false);
    const [open, setOpen] = useState(true);
    const [exp, setexp] = useState({
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'neutral': 0,
        'sad': 0,
        'surprise': 0,
    })

    const handleAccept = () => {
        setOpen(false);
        setSect3(false);
        setSect1(true);
        setRecord(RecordState.START)
        console.log(record);

    };



    const handleRepearQuestionClick = () => {
        vidref.current.load();
        vidref.current.play();


    }
    const videoConstraints = {
        width: 200,
        height: 200,
        facingMode: "user"
    };


    const handleNextQuestionClick = () => {
        const nextQuestion = currentVideo + 1;
        if (nextQuestion < gen_quest.length) {
            setCurrentVideo(nextQuestion);
            if (vidref.current) {
                vidref.current.load();
                vidref.current.play();
            }
        }
        else {
            setSect1(false);
            setSect2(true);

        }
    }

    const handleAnswerOptionClick = (score) => {

        setTotalScore(Totalscore + score);

        const nextQuestion = currentQuestion + 1;
        if (nextQuestion < questions.length) {
            setCurrentQuestion(nextQuestion);
        }


    };

    const handleExitseq = () => {
        setRecord(RecordState.STOP)
    };
    const onData = (audioData) => {
        console.log('chunk of real-time data is: ', audioData);
    }
    function blobToBase64(blob) {
        return new Promise((resolve, _) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
    }

    const onStop = (audioData) => {
        blobToBase64(audioData.blob)
            .then(res => {
                console.log(res)
                axios.post("http://127.0.0.1:5000/classify_dep", { data: res }, {
                    headers: {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,PATCH,OPTIONS',
                    }
                })
                    .then(res => {
                        console.log(res.data)
                    })

            })
    };



    const detect = async (net) => {
        if (
            typeof webcamref.current !== "undefined" &&
            webcamref.current !== null &&
            webcamref.current.video.readyState === 4
        ) {
            // Get Video Properties
            const video = webcamref.current.video;
            const videoWidth = webcamref.current.video.videoWidth;
            const videoHeight = webcamref.current.video.videoHeight;

            // Set video width
            webcamref.current.video.width = videoWidth;
            webcamref.current.video.height = videoHeight;

            // Set canvas width
            canvasRef.current.width = videoWidth;
            canvasRef.current.height = videoHeight;
            const face = await net.estimateFaces({ input: video });
            console.log(face);

            // Get canvas context
            const ctx = canvasRef.current.getContext("2d");
            requestAnimationFrame(() => { drawMesh(face, ctx) });
        }
    };

    const runFacemesh = async () => {
        const net = await facemesh.load(facemesh.SupportedPackages.mediapipeFacemesh);
        setInterval(() => {
            detect(net);
        }, 10);
    };

    React.useEffect(() => { runFacemesh() }, []);


    React.useEffect(() => {
        vidref.current.play();
    })

    // <video autoPlay className={classes.videoDim} ref={vidref}>
    //     <source src={data[count]['src']} type="video/mp4" />
    // </video>

    React.useEffect(() => {
        setTimeout(() => {
            const data = webcamref.current.getScreenshot({ width: 256, height: 256 })
            axios.post("http://127.0.0.1:5000/classify_fer", { data }, {
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,PATCH,OPTIONS',
                }
            })
                .then(res => {
                    // alert(JSON.stringify(res.data))
                    console.log(res.data)
                    if (res.data.success) {
                        exp[res.data.exp] = exp[res.data.exp]++
                        setexp(exp);
                    }
                })
        }, [4000])
    })

    return (

        <Grid container className={classes.root}>


            <Grid container direction="row" justifyContent="space-between">

                <Grid item xs={6} className={classes.webcam}>
                    <Webcam ref={webcamref} videoConstraints={videoConstraints} height={200} width={200} screenshotFormat="image/jpeg"
                        mirrored
                        screenshotQuality={1} />
                    <canvas
                        ref={canvasRef}
                        style={{
                            position: "absolute",

                            left: 0,
                            right: 0,
                            textAlign: "center",
                            zindex: 9,
                            width: 200,
                            height: 200,
                        }}
                    />
                    <AudioReactRecorder canvasWidth={200} canvasHeight={50} backgroundColor="rgb(255,255,255)" foregroundColor="rgb(28, 184, 228)" state={record} onStop={onStop} />

                </Grid>
                {/* <Grid item xs={5}>
                </Grid> */}
                <Grid >
                    <Grid item xs={6} className={classes.BtnPlacement}>

                        <Tooltip placement="left" title="Repeat Question">
                            <IconButton className={classes.Btn} onClick={() => handleRepearQuestionClick()}>
                                <ReplaySharpIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                    <Grid item xs={6} className={classes.BtnPlacement}>
                        <Tooltip placement="left" title="Next Question">
                            <IconButton className={classes.Btn} onClick={() => handleNextQuestionClick()} >
                                <SkipNextSharpIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                    <Grid item xs={6} className={classes.BtnPlacement}>
                        <Tooltip placement="left" title="End Session">
                            <IconButton className={classes.Btn} onClick={() => handleExitseq()} >
                                <HighlightOffSharpIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                </Grid>
            </Grid>

            <Grid item xs={12}>
            </Grid>
            <Grid item xs={12}>
            </Grid>
            <Grid item xs={12}>
            </Grid>
            <Grid item xs={12}>
            </Grid>

            <Grid item xs={12} container
                direction="row"
                justifyContent="center"
                alignItems="center" >

                {sect1 ? (

                    // Generic Questions 
                    <div className={classes.quest_grid}>
                        <video ref={vidref} className={classes.videoDim}>
                            <source src={gen_quest[currentVideo].src} type="video/mp4" />
                        </video>

                        <div direction="row"
                            justifyContent="center"
                            alignItems="center" className={classes.question_sect}>
                            <div className={classes.question_txt}>{gen_quest[currentVideo].questiontext}</div>
                        </div>

                    </div>
                ) : (sect2 ? (

                    // PHQ-8 Questions
                    <>
                            <video ref={vidref} className={classes.videoDim}>
                                <source src={gen_quest[currentVideo].src} type="video/mp4" />
                            </video>
                            <div className={classes.quest_grid}>

                                <div direction="row"
                                    justifyContent="center"
                                    alignItems="center" className={classes.question_sect}>
                                    <div className={classes.question_count}>
                                        <span>Question {currentQuestion + 1}</span>/{questions.length}
                                    </div>
                                    <div className={classes.question_txt}>{questions[currentQuestion].questionText}</div>
                                </div>
                                <div className={classes.answer_sect}>
                                    {questions[currentQuestion].answerOptions.map((answerOption) => (
                                        <Button className={classes.Btn_quest} onClick={() => handleAnswerOptionClick(answerOption.score)}>{answerOption.answerText}</Button>

                                    ))}

                                </div>
                            </div>

                        </>

                    ) : (sect3 ? (
                        // Start Session
                        <div direction="row"
                            justifyContent="center"
                            alignItems="center" >
                            <video autoPlay ref={vidref} muted className={classes.videoDim}>
                                <source src={greeting} type="video/mp4" />
                            </video>

                            <Dialog
                                open={open}
                                disableEscapeKeyDown={true}
                                onClose={handleAccept}
                                aria-labelledby="alert-dialog-title"
                                aria-describedby="alert-dialog-description"
                            >
                                <DialogTitle id="alert-dialog-title">{"Hi! I am EMILY."}</DialogTitle>
                                <DialogContent>
                                    <DialogContentText id="alert-dialog-description">
                                        Pakistan's first digital human and your personal mental health doctor.
                                        Its nice to meet you.
                                        Shall we start the session?
                                    </DialogContentText>
                                </DialogContent>
                                <DialogActions>
                                    <Button onClick={handleAccept} color="primary" autoFocus>
                                        Yes
                                    </Button>
                                    <Button onClick={() => history.push('/')} color="primary" >
                                        No
                                    </Button>
                                </DialogActions>
                            </Dialog>
                    </div>
                        ) : (
                            <>
                                <div>


                                    </div>

                                </>


                )))}

            </Grid>

        </Grid>
    )
}


