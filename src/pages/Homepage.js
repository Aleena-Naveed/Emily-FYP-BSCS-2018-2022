import React from "react";
import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";
import Button from '@material-ui/core/Button';
import { Typography } from "@material-ui/core";
import Logo from '../emily/emily white logo-01.png'
import Typist from 'react-typist';
import "../animation.css";
import Em_hp from "../animations/emily_sp.mp4";



const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,

        minHeight: "100vh",
        overflow: "hidden"
    },


    videoDim: {
        position: 'absolute',
        right: '0',
        bottom: '0',
        minWidth: '100vw',
        height: '100vh',
        objectFit: "cover",
        zIndex: '-2'

    },
    homeText: {
        color: "#fff",
        fontWeight: 'bold',
        padding: "5px 10px 0px 10px",

        "&:hover": {
            color: "#3585da",
        }


    },
    BtnPlacement: {
        marginBottom: "2%",
        marginRight: '2%',

    },
    Btn: {
        borderRadius: 5,
        background: "#fff",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        fontWeight: 'bold',
        color: "#3585da",
        marginTop: 10,
        marginBottom: 5,
        marginRight: 5
    },
    BtnContained: {
        "&:hover": {
            background: "rgba(53,133,218,0.2)",
            color: "#3585da",
        }
    },
    BtnText: {
        fontWeight: 'bold',
        color: "#fff",
        fontSize: 16,
    },
    background_hp: {
        boxShadow: '0 4px 10px 0 rgba(0,0,0,.25)',
    },
    typo: {
        color: "#fff",
        paddingLeft: "20px",
        paddingTop: "5%",

    }
}))


export default function Homepage() {
    const classes = useStyles();
    return (

        <Grid className={classes.root}>
            <Grid item xs={12}>
                <video autoPlay muted className={classes.videoDim}>
                    <source src={Em_hp} type="video/mp4" />
                </video>
            </Grid>
            <Grid>
                <Grid container direction="row" justifyContent="space-between" alignItems="center" xs={12}>
                    <Grid item xs={10}>
                        <img src={Logo} alt="Emily Logo" style={{ width: '80px', paddingTop: '10px', paddingLeft: '20px' }} />
                    </Grid>
                    <Grid item xs={1}>
                        <Typography variant="body1" className={classes.homeText}>HELP</Typography>
                    </Grid>
                    <Grid item xs={1}>

                        <Button

                            variant="contained" className={classes.Btn} classes={{
                                contained: classes.contained
                            }} href="/login"
                        >
                            login
                        </Button>
                    </Grid>
                </Grid>

            </Grid>
            <Grid container className={classes.typo}>
                <Typist>
                    <Typist.Delay ms={500} />
                    <Typography variant="h1">Hi!</Typography>
                    <Typography variant="h2">I am Emily</Typography>
                    <Typography variant="h2">Pakistan's First Digital Human</Typography>
                </Typist>
            </Grid>
            <Grid container>
                <Grid item sx={12} className={'fadein'}>
                    <Button
                        fullWidth
                        variant="contained" className={classes.Btn} classes={{
                            contained: classes.contained
                        }} href="/signup"
                    >
                        Register Now !
                    </Button>
                </Grid>

            </Grid>
        </Grid>
    )
}