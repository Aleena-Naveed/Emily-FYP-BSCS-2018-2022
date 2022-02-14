import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import clsx from 'clsx';

import logo from '../emily/emily white logo-01.png';
import dashboard from '../emily/dashboard.svg';
import report from '../emily/report.svg';
import logouticn from '../emily/logout.svg';
import setting from '../emily/setting.svg';
import user from '../emily/user.svg';
import drawerOpen from '../emily/drawerIconOpen.svg';
import drawerClose from '../emily/drawerIconClose.svg';

import { Grid, Typography } from '@material-ui/core';
import { useHistory } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { logout } from "../pages/stateSlice";
const drawerWidth = 220;

const useStyles = makeStyles((theme) => ({
    '@font-face': {
        fontFamily: 'Montserrat, sans-serif'
    },
    root: {
        display: "flex",
    },
    appBar: {
        zIndex: theme.zIndex.drawer + 1,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
    },
    appBarShift: {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    },
    menuButton: {
        marginRight: 36,
    },
    hide: {
        display: "none",
    },
    drawer: {
        flexShrink: 0,
        whiteSpace: "nowrap",
        dropShadow: "6px 6px 10px rgba(0, 0, 0, 0.16)"
    },
    drawerClose: {
        background: "linear-gradient(#1061b0 0%, #003c72 100%)",
        color: "#fff",
        transition: theme.transitions.create("width", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
        overflow: "hidden",
        width: theme.spacing(8) + 1,
        [theme.breakpoints.up("sm")]: {
            width: theme.spacing(12) + 1,
        },
    },
    drawerOpen: {
        width: drawerWidth,
        background: "linear-gradient(#1061b0 0%, #003c72 100%)",
        color: "#fff",
        transition: theme.transitions.create("width", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
        overflow: "hidden",
    },
    toolbar: {
        backgroundColor: '#3585da',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing(0, 1),
        // necessary for content to be below app bar
        ...theme.mixins.toolbar,
    },
    imageIcon: {
        width: 40,
        height: 40,
        marginRight: 5,
        marginLeft: 10,
    },
    listItemTextStyle: {
        fontSize: 16,
        fontWeight: 'bold',
    },
    listItemLogoOpen: {
        justifyContent: 'center',
        alignItems: 'center',
        margin: 'auto',
        width: 85,
        padding: 0,
    },
    listItemLogoClose: {
        justifyContent: 'flex-start',
        alignItems: 'center',
        width: 80,
        margin: 0,
        padding: 0,
    },
    logoListItem: {
        margin: 0,
        padding: 0,
        marginTop: 20,
        marginBottom: 40,
    },
    dividerStyle: {
        height: 3,
        background: 'rgba(53, 133, 218, 0.4)',
        borderRadius: 15,
        margin: 'auto',
        marginTop: 25,
        marginBottom: 25,
    },
    dividerStyleClose: {
        height: 3,
        background: 'rgba(53, 133, 218, 0.4)',
        margin: 'auto',
        borderRadius: 15,
        marginTop: 25,
        marginBottom: 25,
        width: 70,
    },
    listItemSpacing: {
        marginTop: 15,
        marginBottom: 15
    }
}));

export default function MainDrawer() {
    const classes = useStyles();
    const [open, setOpen] = React.useState(false);

    const handleDrawer = () => {
        setOpen(!open);
    };
    const history = useHistory();
    const dispatch = useDispatch();
    const name = useSelector((state) => state.states.name)
    const id = useSelector((state) => state.states.id)

    return (
        <div className={classes.root}>
            <Grid container>
                <Drawer
                    variant="permanent"
                    className={clsx(classes.drawer, {
                        [classes.drawerOpen]: open,
                        [classes.drawerClose]: !open,
                    })}
                    classes={{
                        paper: clsx({
                            [classes.drawerOpen]: open,
                            [classes.drawerClose]: !open,
                        }),
                    }}>
                    <Grid
                        item
                        className={classes.toolbar}
                        onClick={handleDrawer}
                    >
                        <img src={open ? drawerOpen : drawerClose}
                            alt="Drawer Icon"
                            style={open ? {
                                width: '25px',
                                height: '20px',
                                marginLeft: '90%',
                            } : {
                                width: '25px',
                                height: '20px',
                            }}
                        />
                    </Grid>
                    <Divider />
                    <Grid container>
                        <List>
                            <ListItem key={1} className={classes.logoListItem}>
                                <img src={logo}
                                    alt='Emily Logo'
                                    className={open ?
                                        classes.listItemLogoOpen :
                                        classes.listItemLogoClose
                                    }
                                />
                            </ListItem>
                            <ListItem button key={2} className={classes.listItemSpacing} onClick={() => history.push('/')}>
                                <ListItemIcon>
                                    <img src={dashboard} alt='User Dashboard' className={classes.imageIcon} />
                                </ListItemIcon>
                                <ListItemText
                                    primary={<Typography variant="body1"
                                        className={clsx(classes.listItemTextStyle, {
                                            [classes.hide]: !open,
                                        })}
                                    >DASHBOARD</Typography>} />
                            </ListItem>
                            <ListItem button key={3} className={classes.listItemSpacing} onClick={() => history.push('/reports')}>
                                <ListItemIcon>
                                    <img src={report} alt='Reports' className={classes.imageIcon} />
                                </ListItemIcon>
                                <ListItemText
                                    primary={<Typography variant="body1"
                                        className={clsx(classes.listItemTextStyle, {
                                            [classes.hide]: !open,
                                        })}
                                    >REPORTS</Typography>}
                                />
                            </ListItem>
                            <ListItem button key={4} className={classes.listItemSpacing} onClick={() => history.push('/settings')}>
                                <ListItemIcon>
                                    <img src={setting} alt='Settings' className={classes.imageIcon} />
                                </ListItemIcon>
                                <ListItemText
                                    primary={<Typography variant="body1"
                                        className={clsx(classes.listItemTextStyle, {
                                            [classes.hide]: !open,
                                        })}
                                    >SETTINGS</Typography>} />
                            </ListItem>
                            <Divider className={open ? classes.dividerStyle : classes.dividerStyleClose} />
                            <ListItem button key={5} className={classes.listItemSpacing}
                                onClick={() => history.push("/profile")}>
                                <ListItemIcon>
                                    <img src={user} alt='User' className={classes.imageIcon} />
                                </ListItemIcon>
                                <ListItemText
                                    primary={<Typography variant="body1"
                                        className={clsx(classes.listItemTextStyle, {
                                            [classes.hide]: !open,
                                        })}
                                    >{name}<br /><Typography variant="caption">PID: {id}</Typography></Typography>} />
                            </ListItem>
                            <Divider className={classes.dividerStyle} />
                            <ListItem button key={6} className={classes.listItemSpacing} onClick={() => dispatch(logout())}>
                                <ListItemIcon>
                                    <img src={logouticn} alt='Logout' className={classes.imageIcon} />
                                </ListItemIcon>
                                <ListItemText
                                    primary={<Typography variant="body1"
                                        className={clsx(classes.listItemTextStyle, {
                                            [classes.hide]: !open,
                                        })}
                                    >LOGOUT</Typography>} />
                            </ListItem>
                        </List>
                    </Grid>
                </Drawer>
            </Grid>
        </div>
    );
}
