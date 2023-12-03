"use client";

import {Resizable} from 're-resizable';

import React, { useState, useEffect, useRef } from 'react'  
import useDragger from "../hooks/useDragger";
import Draggable from 'react-draggable';

type cardProps = {
    url: string
};

const Card: React.FC<cardProps> = ({ url })=> {
  
  useDragger(`${url}`);

  const [x, setX]= useState(0)
  const [y, setY]= useState(0)
  const [w, setW]= useState(100)
  const [h, setH]= useState(100)

  const divRef:any = useRef(null);

    useEffect(() => {
        const handleMove = () => {
            if (divRef.current) {
                const rect = divRef.current.getBoundingClientRect();
                // console.log('Element width:', rect.width);
                // console.log('Element height:', rect.height);
                // console.log('Element position X:', rect.left);
                // console.log('Element position Y:', rect.top);
                setX(rect.left)
                setY(rect.top)
            }
        };

        handleMove(); // Initial position

        // Event listeners for moves or updates
        window.addEventListener('resize', handleMove);
        document.addEventListener('mousemove', handleMove);

        return () => {
            window.removeEventListener('resize', handleMove);
            document.removeEventListener('mousemove', handleMove);
        };
    }, []);



    useEffect(() => {
        console.log('Current position X:', x);
        console.log('Current position Y:', y);
    }, [x, y]);

    useEffect(() => {
        console.log('Current width:', w);
        console.log('Current height:', h);
    }, [w, h]);

    function handleStop(e: any, dragElement: any) {
        setX(dragElement.x);
        setY(dragElement.y);
    }

    function handleResizeStop(e: any, direction: any, ref: any, d: any) {
        console.log("hello")
        setW(ref.clientWidth);
        setH(ref.clientHeight);
    }

  return <div id={url} className="object" ref={divRef}>
            <Resizable
                defaultSize={{
                    width: 200,
                    height: 200,
                }}
                style={{
                    background: `url(${url})`,
                    backgroundSize: 'contain',
                    backgroundRepeat: 'no-repeat'
                }}
                onResizeStop={handleResizeStop}
                bounds="window"
                >
            </Resizable>
    </div>
};

export default Card;