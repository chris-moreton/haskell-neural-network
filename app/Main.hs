module Main where

import Time
import Lib
import Data.Time.Clock (getCurrentTime, UTCTime, NominalDiffTime)
import Data.Time.Format (defaultTimeLocale, formatTime)
import Data.Foldable (find)
import Control.Monad
import Control.Monad.IO.Class
import Data.IORef
import Graphics.UI.Gtk hiding (Action, backspace)
import Display
import Api
import Database.MySQL.Simple
import Database
import Control.Concurrent.Async.Timer
import Control.Concurrent (forkIO,  forkOS, threadDelay)
import Control.Monad.Trans (liftIO)
import System.Glib.UTFString
import Update
import LayoutConstants
import GatherData
import Data.List (sortBy)
import Data.Maybe
import qualified Data.Map as Map

main :: IO ()
main = do
  void initGUI
  showWindow

sortCoinData :: CoinData -> CoinData -> Ordering
sortCoinData a b
  | fromJust (Map.lookup 15 (priceChangeMap a)) < fromJust (Map.lookup 15 (priceChangeMap b))  = GT
  | fromJust (Map.lookup 15 (priceChangeMap a)) >= fromJust (Map.lookup 15 (priceChangeMap b))  = LT

ignore :: IO a -> IO ()
ignore ioAction = do
  _ <- ioAction
  return ()

threadLoop :: Int -> IO () -> IO ()
threadLoop delay ioAction = do
    forkIO $ forever $ do
      threadsEnter         -- Acquire the global Gtk lock
      ioAction             -- Perform Gtk interaction like update widget
      threadsLeave         -- Release the global Gtk lock
      threadDelay delay    -- Delay in us
    return ()

gridLoop :: Grid -> Connection -> IO ()
gridLoop grid conn =
    threadLoop 10000000 $ do
      coinData <- gatherData conn
      let sortedCoinData = sortBy sortCoinData coinData
      updateCoinGrid grid sortedCoinData

showWindow :: IO ()
showWindow = do
    window <- windowNew
    set window [ windowTitle         := ("Otter Board" :: String)
               , containerBorderWidth := 10 ]

    grid <- coinGrid
    containerAdd window grid
    widgetModifyBg window StateNormal (Color 0 0 0)

    c <- connectInfo
    conn <- connect c

    gridLoop grid conn

    window `on` deleteEvent $ do
      liftIO mainQuit
      return False

    widgetShowAll window
    mainGUI
